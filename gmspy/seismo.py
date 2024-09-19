from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table
from scipy.integrate import cumulative_trapezoid, trapezoid

from ._const_duct_spec import const_duct_spec
from ._elas_resp_spec import elas_resp_spec
from ._fou_pow_spec import fou_pow_spec
from ._freq_filt import freq_filt


class SeismoGM:
    """A class for computing various intensity measures (IMs) of ground motions,
    as well as elastic response spectra, constant ductility response spectra,
    and spectrum-related intensity measures.

    See the following references for details on IMs:

    * [1] Hariri-Ardebili M A, Saouma V E. Probabilistic seismic demand model and optimal intensity measure
    for concrete dams[J]. Structural Safety, 2016, 59: 67-85.
    .. _DOI: https://doi.org/10.1016/j.strusafe.2015.12.001
    * [2] Yan Y, Xia Y, Yang J, et al. Optimal selection of scalar and vector-valued seismic intensity
    measures based on Gaussian Process Regression[J]. Soil Dynamics and Earthquake Engineering,
    2022, 152: 106961.
    .. _DOI: https://doi.org/10.1016/j.soildyn.2021.106961

    Parameters
    -----------
    dt: float
        The size of the time step of the input acceleration time history.
    acc: 1D ArrayLike
        The acceleration time history.
    unit: str, optional, one of ('g', 'm', 'cm', 'mm', 'in', 'ft')
        Unit of input acceleration time-history, by default "g", ignoring time unit /s2.

    """

    def __init__(self, dt: float, acc: Union[list, tuple, np.ndarray], unit: str = "g") -> None:
        self.dt = dt
        self.acc = np.array(acc)
        if np.abs(self.acc[0]) > 1e-5:
            self.acc = np.insert(self.acc, 0, 0)
        self.nacc = len(self.acc)
        self.time = np.arange(self.nacc) * dt

        self.disp = None
        self.vel = None

        # Arias IMs
        self.Arias = None
        self.AriasSeries = None
        self.AriasPercent = None
        # Tsp
        self.Tsp = None
        self.Spec_sp = None

        # default units
        self.acc_unit = unit
        self.vel_unit = None
        self.disp_unit = None
        self.acc_factor = None
        self.vel_factor = None
        self.disp_factor = None
        self.unit_factors = {
            "g-g": 1.0,
            "g-m": 9.81,
            "g-cm": 981,
            "g-mm": 9810,
            "g-in": 9.81 * 39.3701,
            "g-ft": 9.81 * 3.28084,
            "m-m": 1,
            "m-cm": 100,
            "m-mm": 1000,
            "m-in": 39.3701,
            "m-ft": 3.28084,
            "m-g": 1 / 9.81,
            "cm-m": 0.01,
            "cm-cm": 1,
            "cm-mm": 10,
            "cm-in": 0.393701,
            "cm-ft": 0.0328084,
            "cm_g": 1 / 981,
            "mm-m": 0.001,
            "mm-cm": 0.1,
            "mm-mm": 1,
            "mm-in": 0.0393701,
            "mm-ft": 0.00328084,
            "mm-g": 1 / 9810,
            "in-m": 0.0254,
            "in-cm": 2.54,
            "in-mm": 25.4,
            "in-in": 1,
            "in-ft": 0.0833333,
            "in-g": 1 / (9.81 * 39.3701),
            "ft-m": 0.3048,
            "ft-cm": 30.48,
            "ft-mm": 304.8,
            "ft-in": 12,
            "ft-ft": 1,
            "ft-g": 1 / (9.81 * 3.28084),
        }
        self.set_units(acc=unit, vel="cm", disp="cm", verbose=False)

        # colors
        self.colors = [
            "#037ef3", "#f85a40", "#00c16e", "#7552cc", "#0cb9c1", "#f48924"
        ]

    def set_units(self,
                  acc: str = "g",
                  vel: str = "cm",
                  disp: str = "cm",
                  verbose: bool = True):
        """Specify the unit of input acceleration time-history and the units of output velocity and displacement.

        Parameters
        ----------
        acc : str, optional, one of ('g', 'm', 'cm', 'mm', 'in', 'ft')
            Unit of input acceleration time-history, by default "g",  ignoring time unit /s2.
        vel : str, optional, one of ('m', 'cm', 'mm', 'in', 'ft')
            Unit of output velocity, by default "cm",  ignoring time unit /s.
        disp : str, optional, one of ('m', 'cm', 'mm', 'in', 'ft')
            Unit of output displacement, by default "cm".
        verbose: bool, default=True
            Print info.
        """
        self.acc_factor = self.unit_factors[f"{self.acc_unit}-{acc}"]
        self.acc_unit = acc
        self.acc *= self.acc_factor
        self.vel = cumulative_trapezoid(self.acc, self.time, initial=0)
        self.disp = cumulative_trapezoid(self.vel, self.time, initial=0)
        self.vel_factor = self.unit_factors[f"{acc}-{vel}"]
        self.disp_factor = self.unit_factors[f"{acc}-{disp}"]
        self.vel_unit = vel
        self.disp_unit = disp
        self.vel *= self.vel_factor
        self.disp *= self.disp_factor
        acc_end = "" if self.acc_unit == "g" else "/s2"
        if verbose:
            print(f"[#0099e5]acc-unit: {self.acc_unit}{acc_end};[/]\n"
                  f"[#ff4c4c]vel-unit；{self.vel_unit}/s;[/]\n"
                  f"[#34bf49]disp-unit: {self.disp_unit}[/]")

    def plot_hist(self):
        acc_unit_end = "" if self.acc_unit == "g" else "/$s^2$"
        vel_unit_end = "/s"
        ylabels = [
            f"acc ({self.acc_unit}{acc_unit_end})",
            f"vel ({self.vel_unit}{vel_unit_end})",
            f"disp ({self.disp_unit})",
        ]
        fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex="all")
        plot_y = [self.acc, self.vel, self.disp]
        for i in range(3):
            ax = axs[i]
            ax.plot(self.time, plot_y[i], lw=1.5, c=self.colors[i])
            ax.set_xlim(np.min(self.time), np.max(self.time))
            ax.grid(False)
            ax.set_ylabel(ylabels[i], fontsize=15)
            ax.tick_params(labelsize=12)
        axs[-1].set_xlabel("Time (s)", fontsize=15)
        plt.show()

    def get_acc(self):
        """return acceleration time-history."""
        return self.acc

    def get_vel(self):
        """return velocity time-history."""
        return self.vel

    def get_disp(self):
        """return displacement time-history."""
        return self.disp

    def get_time(self):
        """return time array."""
        return self.time

    def get_time_hists(self):
        """return acceleration, velocity, and displacement time history.

        Returns
        --------
        A length 3 tuple of acceleration, velocity, and displacement time histories.
        """
        return self.acc, self.vel, self.disp

    def get_truncate_hists(self, lower: float = 0.05, upper: float = 0.95):
        """Returns the truncated time-histories for lower-upper Arias intensity.

        Parameters
        -----------
        lower : float, optional, default=0.05
            Lower limit of truncation.
        upper: float, optional, default=0.95
            Upper limit of truncation.

        Returns
        -------
        (acc, vel, disp): 1D Array like.
        """
        if self.Arias is None:
            _ = self.get_ia()
        Arias = self.Arias
        series = self.AriasSeries
        # elements of the time vector which are within the significant duration
        idx = (series >= lower * Arias) & (series <= upper * Arias)
        return self.acc[idx], self.vel[idx], self.disp[idx]

    def get_ims(self, display_results: bool = False):
        """return various IMs independent of response spectra.

        display_results: bool
            If True, display the results.

        Returns
        -------
        A dict of IMs.

        * PGA: peak ground acceleration;
        * PGV: peak ground velocity;
        * PGD: peak ground displacement;
        * V_A: PGV/PGA;
        * D_V: PGD/PGV;
        * EDA: effective design acceleration;
        * Ia: Arias intensity;
        * Ima: Modified Arias intensity;
        * MIV: Maximum Incremental Velocity;
        * Arms,Vrms,Drms: Root-mean-square of acceleration, velocity, and displacement;
        * Pa,Pv,Pd: Housner earthquake power index of acceleration, velocity, and displacement;
        * Ra,Rv,Rd: Riddell index of acceleration, velocity, and displacement;
        * SED: Specific Energy Density;
        * If: Fajfar index;
        * Ic: Park-Ang Index, i.e., characteristic intensity.
        * Icm: Cosenza–Manfredi Intensity;
        * CAV,CAD,CAI: Cumulative Absolute Velocity，Displacement and Impetus;
        * CAVstd: Standardized Cumulative Absolute Velocity;
        * Ip: Impulsivity Index;
        * Tsig_5_95: 5%-95% Arias intensity duration;
        * Tsig_5_75: 5%-75% Arias intensity duration;
        * Tbd: Bracketed duration;
        * Tud: Uniform duration.

        """
        pga = self.get_pga()
        pgv = self.get_pgv()
        pgd = self.get_pgd()
        v_a = self.get_v_a()
        d_v = self.get_d_v()
        eda = self.get_eda()
        arias = self.get_ia()
        arias_corr = self.get_ima()
        miv = self.get_miv()
        arms, vrms, drms = self.get_rms()
        pa, pv, pd = self.get_pavd()
        ra, rv, rd = self.get_ravd()
        sed = self.get_sed()
        fi = self.get_if()
        ic = self.get_ic()
        icm = self.get_icm()
        cav, cad, cai = self.get_cavdi()
        cavstd = self.get_cavstd()
        ip = self.get_ip()
        tsig_5_95, _ = self.get_t_5_95()
        tsig_5_75, _ = self.get_t_5_75()
        t_bd = self.get_brac_td()
        t_ud = self.get_unif_td()
        output = dict(
            PGA=pga,
            PGV=pgv,
            PGD=pgd,
            V_A=v_a,
            D_V=d_v,
            EDA=eda,
            Ia=arias,
            Ima=arias_corr,
            MIV=miv,
            Arms=arms,
            Vrms=vrms,
            Drms=drms,
            Pa=pa,
            Pv=pv,
            Pd=pd,
            Ra=ra,
            Rv=rv,
            Rd=rd,
            SED=sed,
            If=fi,
            Ic=ic,
            Icm=icm,
            CAV=cav,
            CAD=cad,
            CAI=cai,
            CAVstd=cavstd,
            Ip=ip,
            Tsig_5_95=tsig_5_95,
            Tsig_5_75=tsig_5_75,
            Tbd=t_bd,
            Tud=t_ud,
        )
        units, IMnames = _get_ims_unit(self)
        if display_results:
            console = Console()
            table = Table(title="")  # IMs Independent of Spectra
            table.add_column("IM", justify="left", style="bold", no_wrap=True)
            table.add_column("Value", justify="center", style="bold")
            table.add_column("Unit", justify="center", style="bold")
            table.add_column("Name", justify="right", style="bold")
            for key, values in output.items():
                # c = next(colors)
                table.add_row(
                    f"[#0099e5]{key}[/]",
                    f"[#ff4c4c]{values:>.3f}[/]",
                    f"[#34bf49]{units[key]}[/]",
                    f"[#0cb9c1]{IMnames[key]}[/]",
                )
            console.print(table)
        return output

    def get_pga(self):
        """return peak ground values of acceleration (PGA)."""
        return np.max(np.abs(self.acc))

    def get_pgv(self):
        """return peak ground values of velocity (PGV)."""
        return np.max(np.abs(self.vel))

    def get_pgd(self):
        """return peak ground values of displacement (PGD)."""
        return np.max(np.abs(self.disp))

    def get_v_a(self):
        """Peak velocity and acceleration ratio (PGV/PGA)."""
        v_a = self.get_pgv() / self.get_pga() / self.vel_factor
        return v_a

    def get_d_v(self):
        """Peak displacement and velocity ratio (PGD/PGV)."""
        d_v = self.get_pgd() / self.disp_factor / (self.get_pgv() /
                                                   self.vel_factor)
        return d_v

    def get_eda(self, freq=9):
        """EDA，Effective design acceleration,
        defined as the peak acceleration value found after lowpass filtering the input time history
        with a cut-off frequency of 9 Hz [Benjamin and Associates, 1988].

        Parameters
        -----------
        freq: float, default=9HZ
            Frequency threshold.
        """
        acc_filter = freq_filt(self.dt,
                               self.acc,
                               ftype="Butterworth",
                               btype="lowpass",
                               order=5,
                               freq1=freq)
        eda = np.max(np.abs(acc_filter))
        return eda

    def get_ia(self):
        """return Arias intensity IA.
        The Arias intensity (IA) is a measure of the strength of a ground motion.
        It determines the intensity of shaking by measuring the acceleration of transient seismic waves.
        It has been found to be a fairly reliable parameter to describe earthquake shaking necessary
        to trigger landslides. It was proposed by Chilean engineer Arturo Arias in 1970.

        It is defined as the time-integral of the square of the ground acceleration:

        .. math::

            I_{A}={\frac{\\pi}{2g}}\\int_{0}^{T_{d}}a(t)^{2}dt} (m/s)

        where g is the acceleration due to gravity and Td is the duration of signal above threshold.
        The Arias Intensity could also alternatively be defined as the sum of all the squared acceleration
        values from seismic strong motion records.
        """
        # time history of Arias Intensity
        acc = self.acc * self.unit_factors[f"{self.acc_unit}-m"]
        series = np.pi / 2 / 9.81 * cumulative_trapezoid(
            acc**2, self.time, initial=0)
        # Total Arias Intensity at the end of the ground motion
        arias = series[-1]
        # time history of the normalized Arias Intensity
        arias_percent = series / arias
        # ----------------------------------------------------
        self.Arias = arias
        self.AriasSeries = series
        self.AriasPercent = arias_percent
        return arias

    def get_ima(self):
        """Return modified Arias intensity Ima [Araya and Saragoni(1980)].
        Ia/(nZero.^2), nZero is the number of zero points in the acceleration time-history per unit time.
        """
        if self.Arias is None:
            _ = self.get_ia()
        arias = self.Arias
        series = self.AriasSeries
        idx_5_95 = (series >= 0.05 * arias) & (series <= 0.95 * arias)
        timed = self.time[idx_5_95]
        accsig = self.acc[idx_5_95]
        y0 = 0
        below = accsig < y0
        above = accsig >= y0
        kth1 = below[0:-1] & above[1:]
        kth2 = above[0:-1] & below[1:]
        kp1 = np.array([*kth1, False])
        kp11 = np.array([False, *kth1])
        kp2 = np.array([False, *kth2])
        kp22 = np.array([*kth2, False])
        timed01 = (
            np.abs(y0 - accsig[kp1]) * np.abs(timed[kp11] - timed[kp1]) /
            np.abs(accsig[kp11] - accsig[kp1]) + timed[kp1])
        timed02 = (
            np.abs(y0 - accsig[kp22]) * np.abs(timed[kp2] - timed[kp22]) /
            np.abs(accsig[kp2] - accsig[kp22]) + timed[kp22])
        timed0 = np.hstack((timed01, timed02))
        timed0 = np.sort(timed0)
        nzero = len(timed0) / (timed[-1] - timed[0] + self.dt)
        arias_m = self.Arias / nzero**2
        return arias_m

    @staticmethod
    def __iv_aid(time1, time2, acc):
        iv = np.zeros(len(time1) - 1)
        for i in range(len(time1) - 1):
            idxzero1 = np.argwhere(np.abs(time2 - time1[i]) <= 1e-8)
            # locate zero point
            idxzero2 = np.argwhere(np.abs(time2 - time1[i + 1]) <= 1e-8)
            idxzero1 = idxzero1[0, 0]
            idxzero2 = idxzero2[0, 0]
            iv[i] = trapezoid(acc[idxzero1:idxzero2 + 1],
                             time2[idxzero1:idxzero2 + 1])
        return iv

    def get_miv(self):
        """Maximum Incremental Velocity [MIV],
        defined as the maximum value of the time integral of the acceleration between all two zero points.
        """
        if self.Arias is None:
            _ = self.get_ia()
        Arias = self.Arias
        series = self.AriasSeries
        idx_5_95 = (series >= 0.05 * Arias) & (series <= 0.95 * Arias)
        timed = self.time[idx_5_95]
        accsigDura = self.acc[idx_5_95]
        y0 = 0
        below = accsigDura < y0
        above = accsigDura >= y0
        kth1 = below[0:-1] & above[1:]
        kth2 = above[0:-1] & below[1:]
        kp1 = np.array([*kth1, False])
        kp11 = np.array([False, *kth1])
        kp2 = np.array([False, *kth2])
        kp22 = np.array([*kth2, False])
        timed01 = (
            np.abs(y0 - accsigDura[kp1]) * np.abs(timed[kp11] - timed[kp1]) /
            np.abs(accsigDura[kp11] - accsigDura[kp1]) + timed[kp1])
        timed02 = (
            np.abs(y0 - accsigDura[kp22]) * np.abs(timed[kp2] - timed[kp22]) /
            np.abs(accsigDura[kp2] - accsigDura[kp22]) + timed[kp22])
        timed0 = np.hstack((timed01, timed02))
        timed0 = np.sort(timed0)
        Y0 = np.zeros(len(timed0)) + y0
        accdnew0 = np.array([*accsigDura, *Y0])
        timedNew1 = np.array([*timed, *timed0])
        idxMIV = np.argsort(timedNew1)
        timedNew = timedNew1[idxMIV]
        accdnew = accdnew0[idxMIV]
        IV = self.__iv_aid(timed0, timedNew, accdnew)
        miv = np.max(np.abs(IV))
        return miv * self.vel_factor

    def get_sed(self):
        """Specific Energy Density (SED)."""
        sed = trapezoid(self.vel**2, self.time)
        return sed

    def get_rms(self):
        """Root-mean-square (RMS) of acceleration, velocity and displacement."""
        Arms = np.sqrt(trapezoid(self.acc**2, self.time) / self.time[-1])
        Vrms = np.sqrt(trapezoid(self.vel**2, self.time) / self.time[-1])
        Drms = np.sqrt(trapezoid(self.disp**2, self.time) / self.time[-1])
        return Arms, Vrms, Drms

    def get_pavd(self):
        """Housner earthquake power index, Pa, Pv, Pd"""
        if self.Arias is None:
            _ = self.get_ia()
        Arias = self.Arias
        series = self.AriasSeries
        idx_5_95 = (series >= 0.05 * Arias) & (series <= 0.95 * Arias)
        timed = self.time[idx_5_95]
        accsigDura = self.acc[idx_5_95]
        Pa = trapezoid(accsigDura**2, timed) / (timed[-1] - timed[0])
        Pv = trapezoid(self.vel[idx_5_95]**2, timed) / (timed[-1] - timed[0])
        Pd = trapezoid(self.disp[idx_5_95]**2, timed) / (timed[-1] - timed[0])
        return Pa, Pv, Pd

    def get_ravd(self):
        """Riddell index，Ra, Rv, Rd"""
        Td_5_95, _ = self.get_t_5_95()
        PGA = self.get_pga()
        PGV = self.get_pgv()
        PGD = self.get_pgd()
        Ra = PGA * Td_5_95**(1 / 3)
        Rv = PGV**(2 / 3) * Td_5_95**(1 / 3)
        Rd = PGD * Td_5_95**(1 / 3)
        return Ra, Rv, Rd

    def get_if(self):
        """Fajfar index."""
        Td_5_95, _ = self.get_t_5_95()
        PGV = self.get_pgv()
        If = PGV * Td_5_95**0.25
        return If

    def get_ic(self):
        """Characteristic Intensity (Ic)."""
        Arms = self.get_rms()[0]
        Ic = Arms**1.5 * np.sqrt(self.time[-1])
        return Ic

    def get_icm(self):
        """Cosenza–Manfredi Intensity."""
        if self.Arias is None:
            _ = self.get_ia()
        PGA = self.get_pga() * self.unit_factors[f"{self.acc_unit}-m"]
        PGV = self.get_pgv() * self.unit_factors[f"{self.vel_unit}-m"]
        Icm = 2 * 9.81 * self.Arias / (np.pi * PGA * PGV)
        return Icm

    def get_cavdi(self):
        """Cumulative Absolute Velocity (CAV) ，Displacement (CAD) and Impetus(CAI)."""
        CAV = trapezoid(np.abs(self.acc), self.time)
        CAD = trapezoid(np.abs(self.vel), self.time)
        CAI = trapezoid(np.abs(self.disp), self.time)
        return CAV * self.vel_factor, CAD, CAI

    def get_cavstd(self):
        """Standardized Cumulative Absolute Velocity (CAVSTD) [Campbell and Bozorgnia 2011]."""
        ts = np.arange(np.floor(self.time[-1]) + 1)
        acc = self.acc / self.unit_factors[f"g-{self.acc_unit}"]
        idxs = []
        for t in ts:
            idx = np.argmin(np.abs(self.time - t))
            idxs.append(idx)
        idxs.append(len(self.time) - 1)
        cavs = []
        for i in range(len(idxs) - 1):
            p = trapezoid(
                np.abs(acc[idxs[i]:idxs[i + 1] + 1]),
                self.time[idxs[i]:idxs[i + 1] + 1],
            )
            pgai = np.max(np.abs(acc[idxs[i]:idxs[i + 1] + 1]))
            a = 0 if pgai - 0.025 < 0 else 1
            cavs.append(a * p)
        cavstd = np.sum(cavs)
        return cavstd

    def get_ip(self):
        """Impulsivity Index (IP) [Panella et al., 2017].
        An indicator of the impulsive character of the ground motion and is calculated as
        the developed length of velocity of the velocity time-series divided by the Peak Ground Velocity.
        """
        PGV = self.get_pgv()
        vel1 = self.vel[0:-1]
        vel2 = self.vel[1:]
        Ldv = np.sum(np.sqrt((vel2 - vel1)**2 + self.dt**2))
        Ip = Ldv / PGV
        return Ip

    def get_t_5_95(self):
        if self.Arias is None:
            _ = self.get_ia()
        Arias = self.Arias
        series = self.AriasSeries
        # elements of the time vector which are within the significant duration
        idx_5_95 = (series >= 0.05 * Arias) & (series <= 0.95 * Arias)
        timed = self.time[idx_5_95]
        t_5_95 = (timed[0], timed[-1])
        Td_5_95 = timed[-1] - timed[0] + self.dt
        return Td_5_95, t_5_95

    def get_t_5_75(self):
        if self.Arias is None:
            _ = self.get_ia()
        Arias = self.Arias
        series = self.AriasSeries
        # elements of the time vector which are within the significant duration
        idx_5_75 = (series >= 0.05 * Arias) & (series <= 0.75 * Arias)
        timed = self.time[idx_5_75]
        t_5_75 = (timed[0], timed[-1])
        Td_5_75 = timed[-1] - timed[0] + self.dt
        return Td_5_75, t_5_75

    def get_brac_td(self):
        """Bracketed duration.
        The total time elapsed between the first and the last excursions of a specified level of acceleration
        (default is 5% of PGA).
        """
        pga_bd = self.get_pga() * 0.05
        above_bd = np.abs(self.acc) >= pga_bd
        timed = self.time[above_bd]
        # t_bd = (timed[0], timed[-1])
        T_bd = timed[-1] - timed[0]
        return T_bd

    def get_unif_td(self):
        """Uniform duration.
        The total time during which the acceleration is larger than a given threshold value (default is 5% of PGA).
        """
        y0 = self.get_pga() * 0.05
        acc = np.abs(self.acc)
        below = acc < y0
        above = acc > y0
        kth1 = below[0:-1] & above[1:]
        kth2 = above[0:-1] & below[1:]
        kp1 = np.array([*kth1, False])
        kp11 = np.array([False, *kth1])
        kp2 = np.array([False, *kth2])
        kp22 = np.array([*kth2, False])
        timed01 = (
            np.abs(y0 - acc[kp1]) * np.abs(self.time[kp11] - self.time[kp1]) /
            np.abs(acc[kp11] - acc[kp1]) + self.time[kp1])
        timed02 = (
            np.abs(y0 - acc[kp22]) * np.abs(self.time[kp2] - self.time[kp22]) /
            np.abs(acc[kp2] - acc[kp22]) + self.time[kp22])
        T_ud = np.sum(timed02 - timed01)
        return T_ud

    # -----------------------------------------------------
    # ---- response spectrum and IMs ----------------------
    # -----------------------------------------------------
    def get_elas_spec(
        self,
        Ts: Union[float, list, np.ndarray],
        damp_ratio: float = 0.05,
        method: str = "nigam_jennings",
        n_jobs: int = 0,
        plot: bool = False,
    ):
        """Computing the Elastic Response Spectrum.

        Parameters
        ----------
        Ts : Union[float, ArrayLike]
            Eigenperiods for which the response spectra are requested.
        damp_ratio : float, optional
            Damping ratio, by default 0.05.
        method: str, default="Nigam_Jennings"
            Linear Dynamic Time-History Analysis method, optional,
            one of ("FFT", "Nigam_Jennings", "Newmark0", "Newmark1"):

            * "FFT"---Fast Fourier Transform;
            * "Nigam_Jennings"---exact solution by interpolating the excitation over each time interval;
            * "Newmark0"---const acceleration Newmark-beta method, gamma=0.5, beta=0.25;
            * "Newmark1"---linear acceleration Newmark-beta method, gamma=0.5, beta=1/6.

        .. note::
           It is recommended to use the “Nigam_Jennings” method as this is exact for linear systems and
           will be accelerated using
           .. _numba.jit: https://numba.readthedocs.io/en/stable/user/jit.html,
           so speed of computation should not be an issue.

        n_jobs : int, optional, by default 0
            * If 0, do not use parallelism.
            * If an integer greater than 0, call ``joblib`` for parallel computing,
            * and the number of cpu cores used is `n_jobs`.
            * If -1, use all cpu cores.
        plot: bool, default=False
            If True, plot spectra.

        Returns
        -------
        output: (len(Ts), 5) ArrayLike.
            Each column is the *pseudo-acceleration spectrum*, *pseudo-velocity spectrum*,
            *acceleration spectrum*, *velocity spectrum* and *displacement spectrum* in turn.
        """
        Ts = np.atleast_1d(Ts)
        output = elas_resp_spec(self.dt,
                                self.acc,
                                Ts,
                                damp_ratio=damp_ratio,
                                method=method,
                                n_jobs=n_jobs)
        output *= np.array(
            [1, self.vel_factor, 1, self.vel_factor, self.disp_factor]
        )
        if plot:
            acc_unit_end = "" if self.acc_unit == "g" else "/$s^2$"
            vel_unit_end = "/s"
            ylabels = [
                f"PSa ({self.acc_unit}{acc_unit_end})",
                f"PSv ({self.vel_unit}{vel_unit_end})",
                f"Sa ({self.acc_unit}{acc_unit_end})",
                f"Sv ({self.vel_unit}{vel_unit_end})",
                f"Sd ({self.disp_unit})",
            ]
            fig, axs = plt.subplots(5, 1, figsize=(9, 15), sharex="all")
            for i in range(5):
                ax = axs[i]
                ax.plot(Ts, output[:, i], lw=1.5, c=self.colors[i])
                ax.set_xlim(0, np.max(Ts))
                ax.grid(False)
                ax.set_ylabel(ylabels[i], fontsize=15)
                ax.tick_params(labelsize=12)
            axs[-1].set_xlabel("Ts (s)", fontsize=15)
            plt.show()
        output = np.array(output)
        if len(output) == 1:
            return output[0]
        else:
            return output

    def get_const_duct_spec(
        self,
        Ts: Union[float, list, np.ndarray],
        harden_ratio: float = 0.02,
        damp_ratio: float = 0.05,
        analy_dt: float = None,
        mu: float = 5,
        niter: int = 100,
        tol: float = 0.01,
        n_jobs: int = 0,
        plot: int = False,
    ) -> np.ndarray:
        """Constant-ductility inelastic spectra.
        See

        * section 7.5 in Anil K. Chopra (DYNAMICS OF STRUCTURES, Fifth Edition, 2020) and
        * the notes "Inelastic Response Spectra" (CEE 541. Structural Dynamics) by Henri P. Gavin.
        * Papazafeiropoulos, George, and Vagelis Plevris."OpenSeismoMatlab: A new open-source software for strong
          ground motion data processing."Heliyon 4.9 (2018): e00784.

        Parameters
        ----------
        Ts : ArrayLike
            Eigen-periods for which the response spectra are requested.
        harden_ratio : float, optional
            Stiffness-hardening ratio, by default 0.02
        damp_ratio : float, optional
            Damping ratio, by default 0.05.
        analy_dt : float, default = None
            Time step for bilinear SDOF response analysis, if None, default=dt.
        mu : float, optional
            Target ductility ratio, by default 5
        niter : int, optional
            Maximum number of iterations, by default 100
        tol : float, optional
            Controls the tolerance of ductility ratio convergence, by default 0.01
        n_jobs : int, optional, by default 0

            * If 0, do not use parallelism.
            * If an integer greater than 0, call ``joblib`` for parallel computing,
            * and the number of cpu cores used is `n_jobs`.
            * If -1, use all cpu cores.

        plot: bool, default=False
            If True, plot spectra.

        Returns
        -------
        Size (len(Ts), 6) numpy array
            Each column corresponds to acceleration Sa, velocity Sv, displacement Sd spectra,
            yield displacement Dy, strength reduction factor Ry, and yield strength factor Cy (1/Ry).
        """
        output = const_duct_spec(
            self.dt,
            self.acc,
            Ts,
            harden_ratio,
            damp_ratio,
            analy_dt,
            mu,
            niter,
            tol,
            n_jobs,
        )
        output *= np.array(
            [1, self.vel_factor, self.disp_factor, self.disp_factor, 1, 1])
        if plot:
            acc_unit_end = "" if self.acc_unit == "g" else "/$s^2$"
            vel_unit_end = "/s"
            ylabels = [
                f"Sa ({self.acc_unit}{acc_unit_end})",
                f"Sv ({self.vel_unit}{vel_unit_end})",
                f"Sd ({self.disp_unit})",
                f"yield displacement\nDy ({self.disp_unit})",
                "strength reduction factor\nRy",
                "yield strength factor\nCy (1/Ry)",
            ]
            fig, axs = plt.subplots(6, 1, figsize=(9, 18), sharex="all")
            for i in range(6):
                ax = axs[i]
                ax.plot(Ts, output[:, i], lw=1.5, c=self.colors[i])
                ax.set_xlim(0, np.max(Ts))
                ax.grid(False)
                ax.set_ylabel(ylabels[i], fontsize=15)
                ax.tick_params(labelsize=12)
            axs[-1].set_xlabel("Ts (s)", fontsize=15)
            plt.show()
        output = np.array(output)
        if len(output) == 1:
            return output[0]
        else:
            return output

    def get_fou_pow_spec(self, plot: bool = False):
        """The Fourier Amplitude Spectrum and the Power Spectrum (or Power Spectral Density Function)
        are computed by means of Fast Fourier Transformation (FFT) of the input time-history.
        The Fourier amplitude spectrum shows how the amplitude of the ground motion is distributed
        with respect to frequency (or period), effectively meaning that the frequency content of
        the given accelerogram can be fully determined. The power spectral density function,
        on the other hand, may be used to estimate the statistical properties of the input ground motion
        and to compute stochastic response using random vibration techniques.

        * Fourier Amplitude is computed as the square root of the sum of the squares of the
          real and imaginary parts of the Fourier transform: SQRT (Re^2+Im^2);
        * Fourier Phase is computed as the angle given by the real and imaginary parts of
          the Fourier transform: ATAN (Re/Im);
        * Power Spectral Amplitude is computed as FourierAmpl^2/(Pi*duration*RmsAcc^2),
          where duration is the time length of the record, RmsAcc is the acceleration RMS and Pi is 3.14159.

        Parameters
        -----------
        plot: bool, default=False
            If True, plot time histories.

        Returns
        --------
        freq: 1D ArrayLike
            Frequency.
        amp: 1D ArrayLike
            Fourier Amplitude.
        phase: 1D ArrayLike
            Fourier Phase.
        pow_amp: 1D ArrayLike
            Power Spectral Amplitude.
        """
        return fou_pow_spec(self.time, self.acc, plot=plot)

    def get_sac(
        self,
        T1: Union[float, list, np.ndarray],
        damp_ratio: float = 0.05,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        """Cordova Intensity.

        Parameters
        ----------
        T1 : Union[float, ArrayLike]
            The 1st order natural period of the structure.
        damp_ratio : float, optional
            Damping ratio, by default 0.05.
        alpha: float, default=0.5
            coefficient.
        beta: float, default=0.5
            coefficient.

        Returns
        -------
        Sac: Union[float, ArrayLike]
            Cordova Intensity.
        """
        if alpha + beta != 1.0:
            raise ValueError("alpha + beta must be 1 !")
        T1 = np.atleast_1d(T1)
        output1 = self.get_elas_spec(T1, damp_ratio)
        output2 = self.get_elas_spec(T1 * 2, damp_ratio)
        output1 = np.atleast_2d(output1)
        output2 = np.atleast_2d(output2)
        sa1 = output1[:, 2]
        sa2 = output2[:, 2]
        sac = sa1**alpha * sa2**beta
        if len(sac) == 1:
            sac = sac[0]
        return sac

    def get_savam(
        self,
        T1: Union[float, list, np.ndarray],
        T2: Union[float, list, np.ndarray],
        T3: Union[float, list, np.ndarray] = None,
        damp_ratio: float = 0.05,
        alpha: float = 1 / 3,
        beta: float = 1 / 3,
        gamma: float = 1 / 3,
    ):
        """Vamvatsikos Intensity.

        Parameters
        ----------
        T1 : Union[float, ArrayLike]
            The 1st order natural period of the structure.
        T2 : Union[float, ArrayLike]
            The 2nd order natural period of the structure.
        T3 : Union[float, ArrayLike], optional, by default None
            If None, default is 2 * T1
        damp_ratio : float, optional
            Damping ratio, by default 0.05.
        alpha: float, default=1/3
            coefficient.
        beta: float, default=1/3
            coefficient.
        gamma: float, default=1/3
            coefficient.

        Returns
        -------
        Sa_vam: Union[float, ArrayLike]
            Vamvatsikos Intensity.
        """
        if alpha + beta + gamma != 1.0:
            raise ValueError("alpha + beta + gamma must be 1 !")
        T1 = np.atleast_1d(T1)
        T2 = np.atleast_1d(T2)
        if len(T1) != len(T2):
            raise ValueError("length of T1 and T2 must same!")
        if T3 is None:
            T3 = 2 * T1
        output1 = self.get_elas_spec(T1, damp_ratio)
        output2 = self.get_elas_spec(T2, damp_ratio)
        output3 = self.get_elas_spec(T3, damp_ratio)
        output1 = np.atleast_2d(output1)
        output2 = np.atleast_2d(output2)
        output3 = np.atleast_2d(output3)
        sa1, sa2, sa3 = output1[:, 2], output2[:, 2], output3[:, 2]
        sa_vam = sa1**alpha * sa2**beta * sa3**gamma
        if len(sa_vam) == 1:
            sa_vam = sa_vam[0]
        return sa_vam

    def get_samp(
        self,
        T1: Union[float, list, np.ndarray],
        T2: Union[float, list, np.ndarray],
        m1: Union[float, list, np.ndarray],
        m2: Union[float, list, np.ndarray],
        damp_ratio: float = 0.05,
    ):
        """Multiple-Period Intensities.

        Parameters
        -----------
        T1 : Union[float, ArrayLike]
            The 1st order natural period of the structure.
        T2 : Union[float, ArrayLike]
            The 2nd order natural period of the structure.
        m1 : Union[float, ArrayLike]
            The 1st order modal participation factor.
        m2 : Union[float, ArrayLike]
            The 2nd order modal participation factor.
        damp_ratio : float, optional
            Damping ratio, by default 0.05.

        Returns
        -------
        Sa_mp: Union[float, ArrayLike]
            Multiple-Period Intensity.
        """
        output1 = self.get_elas_spec(T1, damp_ratio)
        output2 = self.get_elas_spec(T2, damp_ratio)
        output1 = np.atleast_2d(output1)
        output2 = np.atleast_2d(output2)
        sa1, sa2 = output1[:, 2], output2[:, 2]
        sa_mp = sa1**(m1 / (m1 + m2)) * sa2**(m2 / (m1 + m2))
        if len(sa_mp) == 1:
            sa_mp = sa_mp[0]
        return sa_mp

    def get_avgsavd(self,
                    Tavg: Union[list, tuple, np.ndarray],
                    damp_ratio: float = 0.05,
                    n_jobs: int = 0):
        """Average Spectral Acceleration, Velocity and Displacement.
        They are computed as the geometric mean.

        Parameters
        ----------
        Tavg : 1D ArrayLike
            Period series used to calculate Average Spectral Acceleration.
        damp_ratio : float, optional
            Damping ratio, by default 0.05.
        n_jobs : int, optional, by default 0
            If 0, do not use parallelism.
            If an integer greater than 0, call ``joblib`` for parallel computing,
            and the number of cpu cores used is `n_jobs`.
            If -1, use all cpu cores.

        Returns
        -------
        Savd_avg: 1D ArrayLike
            Average Spectral Acceleration.
            Each element is in the order of acceleration, velocity and displacement.
        """
        n = len(Tavg)
        output = self.get_elas_spec(Tavg, damp_ratio, n_jobs=n_jobs)
        output = np.atleast_2d(output)
        output = output[:, 2:]
        savd_avg = np.exp(np.sum(np.log(output), axis=0) / n)
        return savd_avg

    def _spectrum_prefit(self, damp_ratio):
        """Response spectrum calculation for calculating spectral intensity."""
        self.Tsp = np.arange(0.0, 4.02, 0.02)
        self.Tsp[0] = 0.001
        output = self.get_elas_spec(self.Tsp, damp_ratio, n_jobs=0)
        self.Spec_sp = output
        return output

    def get_savdp(self, damp_ratio: float = 0.05):
        """The peak of the response spectra.

        Parameters
        ----------
        damp_ratio : float, optional
            Damping ratio, by default 0.05.

        Returns
        -------
        Savd_p: 1D ArrayLike
            Each element is in the order of acceleration, velocity and displacement.
        """
        if self.Tsp is None:
            output = self._spectrum_prefit(damp_ratio)
        else:
            output = self.Spec_sp
        Savd_p = np.max(np.abs(output[:, 2:]), axis=0)
        return Savd_p

    def get_avdsi(self, damp_ratio: float = 0.05):
        """Acceleration (ASI)，Velocity (VSI) and Displacement(DSI) Spectrum Intensity.

        Parameters
        ----------
        damp_ratio : float, optional
            Damping ratio, by default 0.05.

        Returns
        -------
        AVD_SI: 1D ArrayLike
            Each element is in the order of acceleration, velocity and displacement.
        """
        if self.Tsp is None:
            output = self._spectrum_prefit(damp_ratio)
        else:
            output = self.Spec_sp
        SIidx1 = np.argwhere(np.abs(self.Tsp - 0.1) <= 1e-8).item()
        SIidx2 = np.argwhere(np.abs(self.Tsp - 0.5) <= 1e-8).item()
        SIidx3 = np.argwhere(np.abs(self.Tsp - 2.5) <= 1e-8).item()
        SIidx4 = np.argwhere(np.abs(self.Tsp - 4.0) <= 1e-8).item()
        Sasp, Svsp, Sdsp = output[:, 2], output[:, 3], output[:, 4]
        ASI = trapezoid(Sasp[SIidx1:SIidx2], self.Tsp[SIidx1:SIidx2])
        VSI = trapezoid(Svsp[SIidx1:SIidx3], self.Tsp[SIidx1:SIidx3])
        DSI = trapezoid(Sdsp[SIidx3:SIidx4], self.Tsp[SIidx3:SIidx4])
        return np.array([ASI, VSI, DSI])

    def get_hsi(self, damp_ratio: float = 0.05):
        """Housner Spectra Intensity (HSI).

        Parameters
        ----------
        damp_ratio : float, optional
            Damping ratio, by default 0.05.

        Returns
        -------
        HSI: float
            Housner Spectra Intensity (HSI)
        """
        if self.Tsp is None:
            output = self._spectrum_prefit(damp_ratio)
        else:
            output = self.Spec_sp
        PSv = output[:, 1]
        HSIidxLow = np.argwhere(np.abs(self.Tsp - 0.1) <= 1e-8).item()
        HSIidxTop = np.argwhere(np.abs(self.Tsp - 2.5) <= 1e-8).item()
        hsi = 1 / 2.4 * trapezoid(PSv[HSIidxLow:HSIidxTop],
                              self.Tsp[HSIidxLow:HSIidxTop])
        return hsi

    def get_epavd(self, damp_ratio: float = 0.05):
        """Effective peak acceleration (EPA), velocity (EPV) and displacement (EPD).

        Parameters
        ----------
        damp_ratio : float, optional
            Damping ratio, by default 0.05.

        Returns
        -------
        EPAVD: 1D ArrayLike
            Each element is in the order of acceleration, velocity and displacement.
        """
        if self.Tsp is None:
            output = self._spectrum_prefit(damp_ratio)
        else:
            output = self.Spec_sp
        Tsp = self.Tsp
        EPidx1 = np.argwhere(np.abs(Tsp - 0.1) <= 1e-8).item()
        EPidx2 = np.argwhere(np.abs(Tsp - 0.5) <= 1e-8).item()
        EPidx3 = np.argwhere(np.abs(Tsp - 0.8) <= 1e-8).item()
        EPidx4 = np.argwhere(np.abs(Tsp - 2.0) <= 1e-8).item()
        EPidx5 = np.argwhere(np.abs(Tsp - 2.5) <= 1e-8).item()
        EPidx6 = np.argwhere(np.abs(Tsp - 4.0) <= 1e-8).item()
        Sa, Sv, Sd = output[:, 2], output[:, 3], output[:, 4]
        EPA = np.sum(Sa[EPidx1:EPidx2]) / (EPidx2 - EPidx1 + 1) / 2.5
        EPV = np.sum(Sv[EPidx3:EPidx4]) / (EPidx4 - EPidx3 + 1) / 2.5
        EPD = np.sum(Sd[EPidx5:EPidx6]) / (EPidx6 - EPidx5 + 1) / 2.5
        return np.array([EPA, EPV, EPD])


def _get_ims_unit(self):
    acc_end = "" if self.acc_unit == "g" else "/s2"
    units = dict(
        PGA=f"{self.acc_unit}{acc_end}",
        PGV=f"{self.vel_unit}/s",
        PGD=f"{self.disp_unit}",
        V_A="s",
        D_V="s",
        EDA=f"{self.acc_unit}{acc_end}",
        Ia="m/s",
        Ima="m/s",
        MIV=f"{self.vel_unit}/s",
        Arms=f"{self.acc_unit}{acc_end}",
        Vrms=f"{self.vel_unit}/s",
        Drms=f"{self.disp_unit}",
        Pa=f"({self.acc_unit}{acc_end})^2",
        Pv=f"({self.vel_unit}/s)^2",
        Pd=f"({self.disp_unit})^2",
        Ra=f"{self.acc_unit}{acc_end}*s^(1/3)",
        Rv=f"({self.vel_unit}/s)^(2/3)*s^(1/3)",
        Rd=f"{self.disp_unit}*s^(1/3)",
        SED=f"{self.vel_unit}2/s",
        If=f"({self.vel_unit}/s)*s^(1/4)",
        Ic=f"({self.acc_unit}{acc_end})^(2/3)*s^(1/2)",
        Icm="--",
        CAV=f"{self.vel_unit}/s",
        CAD=f"{self.vel_unit}",
        CAI=f"{self.disp_unit}*s",
        CAVstd="g*s",
        Ip="--",
        Tsig_5_95="s",
        Tsig_5_75="s",
        Tbd="s",
        Tud="s",
    )
    name = dict(
        PGA="Peak ground acceleration",
        PGV="Peak ground velocity",
        PGD="Peak ground displacement",
        V_A="PGV/PGA",
        D_V="PGD/PGV",
        EDA="Effective Design Acceleration ",
        Ia="Arias Intensity",
        Ima="Modified Arias Intensity",
        MIV="Maximum Incremental Velocity",
        Arms="Root-mean-square of acceleration",
        Vrms="Root-mean-square of velocity",
        Drms="Root-mean-square of displacement",
        Pa="Housner earthquake power index of acceleration",
        Pv="Housner earthquake power index of velocity",
        Pd="Housner earthquake power index of displacement",
        Ra="Riddell index of acceleration",
        Rv="Riddell index of velocity",
        Rd="Riddell index of displacement",
        SED="Specific Energy Density",
        If="Fajfar index",
        Ic="Characteristic Intensity",
        Icm="Cosenza–Manfredi Intensity",
        CAV="Cumulative Absolute Velocity",
        CAD="Cumulative Absolute Displacement",
        CAI="Cumulative Absolute Impetus",
        CAVstd="tandardized Cumulative Absolute Velocity",
        Ip="Impulsivity Index",
        Tsig_5_95=r"5%-95% Arias intensity duration",
        Tsig_5_75=r"5%-75% Arias intensity duration",
        Tbd="Bracketed duration",
        Tud="Uniform duration",
    )
    return units, name


# def _plot_comb_spec(Ts, Sa, Sv, Sd):
#     ticks = []
#     for i in np.arange(-5, 5.1, 1):
#         for j in range(1, 10):
#             ticks.append(10 ** i * j)
#     xlim = (np.min(Ts), np.max(Ts))
#     ylim = (np.min(Sv) / 1.5, np.max(Sv) * 1.5)
#     fig, ax = plt.subplots(figsize=(15, 10))
#     ax.loglog(Ts, Sv, lw=3, c='#0504aa', zorder=10)
#     return None
