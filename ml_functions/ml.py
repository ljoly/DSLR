def normalizeData(topic, minV, maxV):
    for i, _ in enumerate(topic):
        topic[i] = (topic[i]-minV)/(maxV-minV)
    return topic


def getStd(topic, mean):
    std = 0
    for mark in topic:
        std += (mark - mean) ** 2
    std /= len(topic) - 1
    std = std ** 0.5
    return std


def getMinMax(topic):
    # Get min and max values
    topic = sorted(topic)
    minV = topic[0]
    maxV = topic[len(topic) - 1]
    return minV, maxV


def getMean(topic):
    mean = 0
    for i in range(len(topic)):
        mean += topic[i]
    mean /= len(topic)
    return mean


def formatData(topic):
    for i, elem in enumerate(topic):
        topic[i] = float(elem)
    return topic

def isFormatted(row):
    for r in row:
        if r == "":
            return False
    return True
