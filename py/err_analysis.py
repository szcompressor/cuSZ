# 23-12-29

from _lrz import *
from _utils import *
from typing import List, Tuple


class MemorySegment:
    def __init__(self, _d: np.ndarray) -> None:
        self.data = _d
        self.minval, self.maxval = extrema(_d)
        self.rng = self.maxval - self.minval

        self.pred_err = None
        self.errquant = None

        self.prequant = None
        self.hist_errquant = []

    def init_sz14(self):
        self.pred_err = np.zeros_like(self.data)
        self.errquant = np.zeros_like(self.data)

    def init_psz(self):
        self.prequant = np.zeros_like(self.data)
        self.errquant = np.zeros_like(self.data)


class OneDataMultipleErrorBound:
    def __init__(
        self, fname: str = None, size=None, dtype: str = "f4", data: np.ndarray = None
    ) -> None:
        ## TODO better be immutable from now on
        self.__data = None

        if data is None:
            print("OneDataMultipleErrorBound: loading data using fname")
            self.__data = load_data(fname, size, dtype)
        else:
            print("OneDataMultipleErrorBound: loading data from the existing")
            self.__data = data

        self.data = MemorySegment(self.__data)

        self.eb_list_input = None
        self.eb_list_final = None

        self.sz14 = MemorySegment(self.__data)
        self.psz = MemorySegment(self.__data)

    def adjust_eb(self, eb_list: List[float], rel2rng=True):
        ## adjust eb according to mode (i.e., if relative_to_range)
        self.eb_list_input = np.array(eb_list)
        if rel2rng:
            self.eb_list_final = self.eb_list_input * self.data.rng
        else:
            self.eb_list_final = self.eb_list_input

    def run_sz14(self, eb_list: List[float], if_rel2rng=True) -> None:
        self.adjust_eb(eb_list, if_rel2rng)
        self.sz14.init_sz14()

        Sz14Approx.lorenzo_predict(self.sz14.data, self.sz14.pred_err)
        for eb in self.eb_list_final:
            Sz14Approx.quantize(
                self.sz14.pred_err, self.sz14.errquant, eb, in_place=False
            )
            emin, emax = extrema(self.sz14.errquant)
            hist = np.histogram(self.sz14.errquant, bins=np.arange(emin, emax + 1))
            self.sz14.hist_errquant.append(hist)

    def run_psz(self, eb_list: List[float], if_rel2rng=True) -> None:
        self.adjust_eb(eb_list, if_rel2rng)
        self.psz.init_psz()

        for eb in self.eb_list_final:
            PszApprox.preceding_quantize(self.psz.data, self.psz.prequant, eb)
            PszApprox.succeeding_lorenzo_predict(self.psz.prequant, self.psz.errquant)

            emin, emax = extrema(self.psz.errquant)
            hist = np.histogram(self.psz.errquant, bins=np.arange(emin, emax + 1))
            self.psz.hist_errquant.append(hist)

    @staticmethod
    def hist_xy(hist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return hist[1][:-1], hist[0]
