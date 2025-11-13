class ContrastiveLoss:
    def __init__(self, margin=1.0):
        self.margin = margin

    def __call__(self, anchor, positive, negative):
        positive_distance = self._euclidean_distance(anchor, positive)
        negative_distance = self._euclidean_distance(anchor, negative)
        loss = max(0, positive_distance - negative_distance + self.margin)
        return loss

    def _euclidean_distance(self, x1, x2):
        return ((x1 - x2) ** 2).sum().sqrt()


class TripletLoss:
    def __init__(self, margin=1.0):
        self.margin = margin

    def __call__(self, anchor, positive, negative):
        positive_distance = self._euclidean_distance(anchor, positive)
        negative_distance = self._euclidean_distance(anchor, negative)
        loss = max(0, positive_distance - negative_distance + self.margin)
        return loss

    def _euclidean_distance(self, x1, x2):
        return ((x1 - x2) ** 2).sum().sqrt()