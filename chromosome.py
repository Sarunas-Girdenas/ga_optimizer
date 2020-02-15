from typing import Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class Chromosome():
    """This is Chrosome type which is immutable
    after instatation.
    """

    chromosome: Dict

    @property
    def length(self):
        """Returns lengths of chromosome
        """

        return len(self.chromosome)

    @property
    def max_weight(self):
        """Returns maximum allowed weight
        of chromosome
        """

        return self.chromosome['max_weight']

    @property
    def weight(self):
        """Returns actual weight
        of chromosome
        """

        return self.chromosome['weight']