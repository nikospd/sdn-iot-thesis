# Simulation for sdn network to evaluate qos-based routing algorithms

# assumptions: numOfGw + numOfHosts >= numOfSwitches
# Future work: multiple QoS, Multiple hosts per priority, changing number of switches

import math
import random
import itertools
import time
import simpy
import re
from utils import plotNodeTopo
from abc import ABC, abstractmethod
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

numOfNodes = 50
numOfGw = 5
numOfSwitches = 15
numOfHosts = 3
evaluationFlag = dict()
admissionControlFlag = False
maxQOccAfter = dict()
maxQOccBefore = dict()
flowFreqValidation = dict()


class Node:
    newId = itertools.count()
    id: int
    hId: str
    x: int
    y: int
    priority: int  # Priority [0,1,2] from high to low
    gwId: [int]
    packetSize: int  # The average length of the packet that this node provides (Bytes)
    routes = []
    period: int  # In milliseconds
    dc: int  # Drop counter for this flow
    drop_flag: bool  # Flag to indicate if it has to drop the next packet or not

    def __init__(self, en, maxX=30, maxY=30, priority=0, pktSize=5000, period=100):
        self.action = None
        self.x = random.randint(-maxX, maxX)
        self.y = random.randint(-maxY, maxY)
        self.priority = priority
        self.id = next(Node.newId)
        self.hId = "n" + str(self.id)
        self.gwId = []
        self.packetSize = pktSize
        self.period = period
        self.env = en
        self.produced_pkt = 0
        self.dc = 0
        self.start()

    def reselectCoordinates(self, maxX=30, maxY=30):
        self.x = random.randint(-maxX, maxX)
        self.y = random.randint(-maxY, maxY)

    def run(self):
        while True:
            new_pkt = Packet(self.id, self.priority, Node.routes[self.id][1:], self.env.now, self.packetSize)

            new_pkt.currentHop = 0
            self.produced_pkt += 1
            if self.dc > 0:
                hosts[self.priority].drop_pkt()
                self.dc -= 1
            else:
                gateways[Node.routes[self.id][0]].add_pkt(new_pkt)
            yield self.env.timeout(np.random.exponential(self.period))

    def drop_pkt(self):
        self.dc += 1

    def start(self):
        self.action = self.env.process(self.run())

    def stop(self):
        self.action = None


class EthConnection:
    newId = itertools.count()
    id: int
    bw: int  # MBs/s
    hId: str

    def __init__(self):
        self.id = next(EthConnection.newId)
        self.hId = "eth" + str(self.id)


class WirelessConnection:
    newId = itertools.count()
    id: int
    bw: int  # MBs/s
    hId: str

    def __init__(self):
        self.id = next(WirelessConnection.newId)
        self.hId = "w" + str(self.id)


class Packet:
    newId = itertools.count()
    id: int
    size: int  # In bytes
    src: int  # The source of the packet (e.x. 2 as n2)
    dst: int  # The destination of the packet (e.x. 1 as h1)
    route: [int]  # The route that this packet will follow (contains the id's of the switches)
    priority: int  # The priority of for the packet
    startTime: int  # The starting time of this packet
    delay: int  # The delay value (ms)
    dropped: bool  # Flag that indicates if this packed dropped or not
    currentHop: int  # The number of the current hop
    pu: []  # Port Utilization array

    def __init__(self, src, dst, route, startTime, size=100):
        self.id = next(Packet.newId)
        self.startTime = startTime
        self.src = src
        self.dst = dst
        self.route = route
        self.size = size
        self.delay = 0
        self.dropped = False
        self.pu = []


class ServiceStation(ABC):
    id: int
    hId: str

    qSize: int
    ts: int  # Serving speed of packets (byte/ms) 100Mbps=>12500Byte/ms
    pktQ: [Packet]  # The actual queue of packets in each station of service
    timeOnService: int  # How much time the first packet is already under service
    maxQ: int  # Keeps the maximum occupation during the simulation
    flowIds: []  # Keeps the ids of the flows that uses this station with their priorities

    def __init__(self, en, queueSize=8, tService=1250):
        self.action = None
        self.pktQ = []
        self.qSize = queueSize
        self.ts = tService
        self.env = en
        self.maxQ = 0
        self.start()
        self.flowIds = []

    def add_pkt(self, pkt):
        # print("[ARR] station %s packet id %d time %d" % (self.hId, pkt.id, self.env.now))
        self.env.timeout(pkt.size / self.ts)  # Simulates needed time to transmit a packet through the eth line
        if 0 < self.qSize <= len(self.pktQ):
            hosts[pkt.dst].drop_pkt()
        else:
            pkt.pu.append({"port": self.hId, "utilization": len(self.pktQ)})
            self.pktQ.append(pkt)
            if len(self.pktQ) > self.maxQ: self.maxQ = len(self.pktQ)


    def remove_pkt(self):
        self.pktQ.pop(0)  # FIFO queue

    def start(self):
        self.action = self.env.process(self.run())

    @abstractmethod
    def findNextDst(self):
        pass

    @abstractmethod
    def run(self):
        pass


class Gateway(ServiceStation):
    newId = itertools.count()
    x: int
    y: int
    R: int

    def __init__(self, en, maxX=25, maxY=20, R=20, rad=False, num=2):
        super().__init__(en)
        self.id = next(Gateway.newId)
        self.hId = "gw" + str(self.id)
        if rad:
            r = self.id * (2 * math.pi / num)
            self.x = maxX * math.sin(r)
            self.y = maxY * math.cos(r)
        else:
            self.x = random.randint(-maxX, maxX)
            self.y = random.randint(-maxY, maxY)
        self.R = R
        self.env = en

    def run(self):
        while True:
            while len(self.pktQ) > 0:
                # print("[GWA] station %s packet id %d time %d" % (self.hId, self.pktQ[0].id, env.now))
                yield self.env.timeout(self.pktQ[0].size / self.ts)
                switches[self.findNextDst()].add_pkt(self.pktQ[0])
                self.remove_pkt()
            yield self.env.timeout(0.5)

    def findNextDst(self):
        return self.pktQ[0].route[0]


class Switch(ServiceStation):
    newId = itertools.count()

    def __init__(self, en):
        super().__init__(en)
        self.id = next(Switch.newId)
        self.hId = "s" + str(self.id)
        self.env = en

    def run(self):
        while True:
            while len(self.pktQ) > 0:
                # print("[SWI] station %s packet id %d time %d" % (self.hId, self.pktQ[0].id, env.now))
                yield self.env.timeout(self.pktQ[0].size / self.ts)
                nextId, is_terminal = self.findNextDst()
                if not is_terminal:
                    self.pktQ[0].currentHop += 1
                    switches[nextId].add_pkt(self.pktQ[0])
                else:
                    hosts[nextId].add_pkt(self.pktQ[0])
                self.remove_pkt()
            yield self.env.timeout(0.5)

    def findNextDst(self):
        currentHop = self.pktQ[0].currentHop
        if currentHop < len(self.pktQ[0].route) - 1:
            return self.pktQ[0].route[currentHop + 1], False
        else:
            return self.pktQ[0].dst, True


class Host(ServiceStation):
    newId = itertools.count()

    def __init__(self, en):
        super().__init__(en)
        self.id = next(Host.newId)
        self.hId = "h" + str(self.id)
        self.delay_times = 0
        self.arrived_pkt = 0
        self.pkt_loss = 0
        self.rDelay = 22 + 2 * self.id  # Delay requirement depend on the priority of the host

    def run(self):
        while True:
            while len(self.pktQ) > 0:
                yield self.env.timeout(self.pktQ[0].size / self.ts)
                self.delay_times += (self.env.now - self.pktQ[0].startTime)
                self.arrived_pkt += 1
                self.pktQ[0].delay = self.env.now - self.pktQ[0].startTime
                if evaluationFlag[self.pktQ[0].src]:
                    RoutingAlgorithm.evaluate_pkt(self.pktQ[0], self.rDelay)
                if admissionControlFlag:
                    RoutingAlgorithm.admission_control(self.pktQ[0], self.rDelay)
                self.remove_pkt()
            yield self.env.timeout(0.5)

    def findNextDst(self):
        return -1, True

    def drop_pkt(self):
        self.pkt_loss += 1


class RoutingAlgorithm:
    recordPathList: {}
    iteration: int

    def __init__(self, en):
        print("init routing algorithm")
        self.action = None
        self.env = en
        RoutingAlgorithm.recordPathList = {}
        RoutingAlgorithm.iteration = 5
        self.start()

    def start(self):
        self.action = self.env.process(self.run())

    ############################
    # Based on the bandwidth
    ############################
    @staticmethod
    def evaluate_pkt(pkt, rDelay):
        if pkt.src + 1 == numOfNodes:
            global admissionControlFlag
            create_route_flow_table()
            admissionControlFlag = True
        cp = (pkt.delay - rDelay) / rDelay
        if cp < 0:
            evaluationFlag[pkt.src] = False
            evaluationFlag[pkt.src + 1] = True
            return
        available_paths = list(map(lambda x: list(map(get_id_from_hid, x[1:-1])), list(nx.all_shortest_paths(G, "n"+str(pkt.src), "h"+str(pkt.dst)))))
        min_max_bw = []
        for path in available_paths:
            max_bw = []
            max_bw.append(flowFreqValidation["gw"+str(path[0])])
            for point in path[1:]:
                max_bw.append(flowFreqValidation["s" + str(point)])
            min_max_bw.append({"route": path, "max_bw": max(max_bw)})
        flowFreqValidation["gw" + str(Node.routes[pkt.src][0])] -= 1
        for point in Node.routes[pkt.src][1:]:
            flowFreqValidation["s" + str(point)] -= 1
        Node.routes[pkt.src] = sorted(min_max_bw, key=lambda i: i["max_bw"])[0]["route"]
        flowFreqValidation["gw" + str(Node.routes[pkt.src][0])] += 1
        for point in Node.routes[pkt.src][1:]:
            flowFreqValidation["s" + str(point)] += 1
        evaluationFlag[pkt.src] = False
        evaluationFlag[pkt.src + 1] = True

    @staticmethod
    def admission_control(pkt, rDelay):
        cp = (pkt.delay - rDelay) / rDelay
        if cp < 0:
            return
        if pkt.dst > 0:
            return
        vp_flows = RoutingAlgorithm.find_vp_flow(pkt, -1)
        if len(vp_flows) == 0:
            vp_flows = RoutingAlgorithm.find_vp_flow(pkt, -2)
        vp_flows = [dict(item, **{"dc": nodes[item["id"]].dc}) for item in vp_flows]
        if len(vp_flows) != 0:
            vf = list(sorted(vp_flows, key=lambda i: i["dc"]))[0]["id"]
            if random.random() < get_dp_from_dc(nodes[vf].dc):
                nodes[vf].drop_pkt()

    @staticmethod
    def find_vp_flow(pkt, pst):
        vp = sorted(pkt.pu[:-1], key=lambda i: i["utilization"])[pst]["port"]
        vp_flows = []
        if get_h_from_hid(vp) == "s":
            vp_flows = switches[get_id_from_hid(vp)].flowIds
        elif get_h_from_hid(vp) == "gw":
            vp_flows = gateways[get_id_from_hid(vp)].flowIds
        vp_flows = list(filter(lambda i: i["priority"] == 2, vp_flows))
        if len(vp_flows) == 0:
            vp_flows = list(filter(lambda i: i["priority"] == 1, vp_flows))
        return vp_flows
    ############################
    # Based on the cost function
    ############################
    # @staticmethod
    # def evaluate_pkt(pkt, rDelay):
    #     cp = (pkt.delay - rDelay) / rDelay
    #     RoutingAlgorithm.recordPathList[pkt.src].append({"route": Node.routes[pkt.src], "cp": cp})
    #     if len(RoutingAlgorithm.recordPathList[pkt.src]) < RoutingAlgorithm.iteration:
    #         RoutingAlgorithm.updateRoute(pkt.src, pkt.dst, len(RoutingAlgorithm.recordPathList[pkt.src]))
    #     else:
    #         RoutingAlgorithm.chooseBestCandidate(RoutingAlgorithm.recordPathList[pkt.src], pkt.src)
    #
    # @staticmethod
    # def chooseBestCandidate(recordPathList, nodeId):
    #     Node.routes[nodeId] = sorted(recordPathList, key=lambda i: i["cp"])[0]["route"]
    #     evaluationFlag[nodeId] = False
    #     evaluationFlag[nodeId + 1] = True
    #
    # @staticmethod
    # def updateRoute(src, dst, iteration):
    #     l = list(nx.all_shortest_paths(G, "n" + str(src), "h" + str(dst)))
    #     if len(l) <= iteration:
    #         route = l[-1]
    #     else:
    #         route = l[iteration]
    #     Node.routes[src] = list(map(get_id_from_hid, route[1:-1]))

    @staticmethod
    def init_recordPathList():
        for n in nodes:
            RoutingAlgorithm.recordPathList[n.id] = []

    def run(self):
        yield self.env.timeout(2 * 60 * 1000)
        global flowFreqValidation
        flowFreqValidation = get_usage()
        global maxQOccBefore
        maxQOccBefore = results_of_simulation()
        # plot_q_uccupation(maxQOccBefore, only_before=True)
        empty_queues()
        print("Start evaluation!")
        yield self.env.timeout(0.1 * 60 * 1000)
        RoutingAlgorithm.init_recordPathList()
        start_evaluation()


def empty_queues():
    for gw in gateways:
        if len(gw.pktQ) > 0:
            gw.pktQ = [gw.pktQ[-1]]
        else:
            gw.pktQ = []
    for h in hosts:
        if len(h.pktQ) > 0:
            h.pktQ = [h.pktQ[-1]]
        else:
            h.pktQ = []
        h.pkt_loss = 0
        h.delay_times = 0
        h.arrived_pkt = 0
    for s in switches:
        s.maxQ = 0
        if len(s.pktQ) > 0:
            s.pktQ = [s.pktQ[-1]]
        else:
            s.pktQ = []


def routeNodes(s: Node, gwL: [Gateway]):
    for gw in gwL:
        dist = math.sqrt((s.x - gw.x) ** 2 + (s.y - gw.y) ** 2)
        if dist <= gw.R:
            s.gwId.append(gw.id)
    if len(s.gwId) == 0:
        s.reselectCoordinates()
        routeNodes(s, gwL)


def increase_if_exist(d, key):
    if d.get(key):
        d[key] += 1
    else:
        d[key] = 1


def get_id_from_hid(hid):
    return int(re.findall('\d+', hid)[0])


def get_h_from_hid(hid):
    return "".join(re.findall('[a-z]', hid))


def get_hid_from_route(route):
    res = []
    res.append("gw" + str(route[0]))
    for r in route[1:]:
        res.append("s"+str(r))
    return res


def get_dp_from_dc(dc):
    if dc == 0:
        return 1
    else:
        return 1/dc


def create_route_flow_table():
    for idx, route in enumerate(Node.routes):
        gateways[route[0]].flowIds.append({"id": idx, "priority": nodes[idx].priority})
        for point in route[1:]:
            switches[point].flowIds.append({"id": idx, "priority": nodes[idx].priority})


def start_evaluation():
    evaluationFlag[0] = True


def stop_evaluation():
    for n in nodes:
        evaluationFlag[n.id] = False


############################
# Calculate and show the results
############################
def results_of_simulation():
    arr_pkt = 0
    for h in hosts:
        arr_pkt += h.arrived_pkt
        print("host: %d average delay: %f | packet loss: %f" % (h.id, h.delay_times / h.arrived_pkt, h.pkt_loss / (h.arrived_pkt + h.pkt_loss)))

    prd_pkt = 0
    for n in nodes:
        prd_pkt += n.produced_pkt
        n.produced_pkt = 0

    print("produced %d | arrived %d | diff %d" % (prd_pkt, arr_pkt, arr_pkt - prd_pkt))

    maxQOcc = dict()
    for s in switches:
        maxQOcc[s.hId] = s.maxQ
    return maxQOcc


def get_usage():
    ############################
    # Plot Diagrams
    ############################
    flowFreq = dict()
    for s in switches:
        flowFreq[s.hId] = 0
    for gw in gateways:
        flowFreq[gw.hId] = 0
    for node_route in Node.routes:
        for p in get_hid_from_route(node_route):
            increase_if_exist(flowFreq, p)
    return flowFreq


def plot_q_uccupation(before, after=None, only_before=False):
    x3 = np.arange(numOfSwitches)
    width = 0.35
    fig3, ax3 = plt.subplots()
    rects1 = ax3.bar(x3 - width / 2, list(before.values())[:numOfSwitches], width, label='Before')
    if not only_before:
        rects2 = ax3.bar(x3 + width / 2, list(after.values())[:numOfSwitches], width, label='After')
        ax3.bar_label(rects2, padding=3)
    ax3.set_title("Max queue occupation")
    ax3.legend()
    ax3.bar_label(rects1, padding=3)
    plt.show()


def plot_diagrams_of_usage(Before, After=None, only_before=False):

    x1 = np.arange(numOfSwitches)
    width = 0.35
    fig1, ax1 = plt.subplots()
    rects1 = ax1.bar(x1 - width / 2, list(Before.values())[:numOfSwitches], width, label='Before')
    ax1.bar_label(rects1, padding=3)
    if not only_before:
        rects2 = ax1.bar(x1 + width / 2, list(After.values())[:numOfSwitches], width, label='After')
        ax1.bar_label(rects2, padding=3)
    ax1.set_title("Switches in-flow usage")
    ax1.legend()
    ax1.set_xticks(x1, list(Before.keys())[:numOfSwitches])
    plt.show()

    x2 = np.arange(numOfGw)
    fig2, ax2 = plt.subplots()
    rects1 = ax2.bar(x2 - width / 2, list(Before.values())[numOfSwitches:], width, label='Before')
    if not only_before:
        rects2 = ax2.bar(x2 + width / 2, list(After.values())[numOfSwitches:], width, label='After')
        ax2.bar_label(rects2, padding=3)
    ax2.set_title("Gateways in-flow usage")
    ax2.legend()
    ax2.bar_label(rects1, padding=3)
    ax2.set_xticks(x2, list(Before.keys())[numOfSwitches:])
    plt.show()


if __name__ == '__main__':
    env = simpy.Environment()
    ############################
    # Create node topology
    ############################
    nodes = []
    gateways = []
    for i in range(numOfNodes):
        nodes.append(Node(env, priority=random.randint(0, 2)))
    for i in range(numOfGw):
        gateways.append(Gateway(env, rad=True, num=numOfGw))
    for node in nodes:
        routeNodes(node, gateways)
    # plotNodeTopo(nodes, gateways)
    ############################
    # Create switch topology
    ############################
    # G = nx.Graph()
    G = nx.DiGraph()
    switches = []
    ethernetConnections = []
    for i in range(numOfSwitches):
        tmpSwitch = Switch(env)
        switches.append(tmpSwitch)
        G.add_node(tmpSwitch.hId, color="red")
    for i in range(numOfSwitches):
        for j in range(numOfSwitches):
            if i == j:
                continue
            if random.random() > 0.8:
                # Create a sparse network
                tmpEth = EthConnection()
                ethernetConnections.append(tmpEth)
                G.add_edge(switches[i].hId, switches[j].hId, color="red", width=2, id=tmpEth.hId)
                G.add_edge(switches[j].hId, switches[i].hId, color="red", width=2, id=tmpEth.hId)
    hosts = []
    for i in range(numOfHosts):
        hosts.append(Host(env))
    ############################
    # Attach gateways and hosts to the network
    ############################
    topoFreq = dict()
    for gateway in gateways:
        topoFreq[gateway.hId] = 0
        G.add_node(gateway.hId, color="blue")
        tmpEth = EthConnection()
        ethernetConnections.append(tmpEth)
        G.add_edge(gateway.hId, switches[gateway.id].hId, color="blue", width=2, id=tmpEth.hId)
    for host in hosts:
        G.add_node(host.hId, color="green")
        for i in range(2):
            tmpEth = EthConnection()
            ethernetConnections.append(tmpEth)
            G.add_edge(switches[2 * host.id + i + numOfGw].hId, host.hId, color="green", width=2, id=tmpEth.hId)
    routes = []
    ############################
    # Plot Network Diagram
    ############################
    net = Network("100%", "100%")
    net.from_nx(G)
    net.show("dsa.html")
    ############################
    # Attach nodes to the gateways
    ############################
    wirelessConnections = []
    for node in nodes:
        G.add_node(node.hId, color="grey")
        for gwId in node.gwId:
            topoFreq["gw" + str(gwId)] += 1  # If I want to plot gw in-topo usage
            tmpWireless = WirelessConnection()
            wirelessConnections.append(tmpWireless)
            G.add_edge(node.hId, gateways[gwId].hId, color="grey", width=2, id=tmpWireless.hId)
    ############################
    # Calculate the flows
    ############################
    for node in nodes:
        node.route = nx.shortest_path(G, node.hId, "h" + str(node.priority))
        switch_route = list(map(get_id_from_hid, node.route[1:-1]))
        routes.append(switch_route)
    Node.routes = routes
    ############################
    # Plot Diagrams
    ############################
    flowFreqBefore = get_usage()
    # plot_diagrams_of_usage(flowFreqBefore, only_before=True)
    ############################
    # Run the simulation
    ############################
    stop_evaluation()
    RoutingAlgorithm(env)
    print("start of simulation")
    env.run(until=4 * 60 * 1000)
    print("end of simulation")
    ############################
    # Plot Diagrams
    ############################
    flowFreqAfter = get_usage()
    plot_diagrams_of_usage(flowFreqBefore, flowFreqAfter)
    maxQOccAfter = results_of_simulation()
    plot_q_uccupation(maxQOccBefore, maxQOccAfter)
