class CompositeProbe:
    def __init__(self, probes):
        self.probes = probes

    def __iter__(self):
        for probe in self.probes:
            yield from probe
