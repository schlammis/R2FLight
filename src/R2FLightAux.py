import numpy as np
import scipy.optimize
from dataclasses import dataclass


def fit_sine_cplx(y, fsamp, fsig, fline=60, Nhars=1, use_hann=True, chunk_periods=0, cache=None):
    """
    Fits a sine wave with a DC offset to the data.

    When chunk_periods > 0, uses two-stage fitting:
      1. Global Hanning-windowed fit removes DC, line harmonics, and f/2 DDS spur.
      2. The signal-only residual is split into chunk_periods-long windows; each chunk
         gets its own [DC, cos, sin, cos/2, sin/2] fit. The array of per-chunk complex
         amplitudes is returned as a 5th element so that eta ratios can be computed
         per chunk before averaging (avoiding amplitude-averaging bias).

    When cache= is supplied (from build_fit_cache), the design-matrix construction and
    inversion are skipped; each call reduces to a matrix–vector multiply.

    Returns:
        4-tuple (complex_amp, fit_vals, errv, rss)                   when chunk_periods == 0
        5-tuple (complex_amp, fit_vals, errv, rss, chunk_amps)       when chunking succeeds
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_chunk_params = 5  # DC, cos(wt), sin(wt), cos(wt/2), sin(wt/2)

    if cache is not None:
        X, w, bg_pinv, rf = cache['X'], cache['w'], cache['bg_pinv'], cache['rf']
        fit_pars   = bg_pinv @ (y * w)
        use_chunks = 'chunk_pinvs' in cache
    else:
        rf  = fsig  / fsamp
        rlf = fline / fsamp
        wt  = 2 * np.pi * np.arange(n) * rf
        wlf = 2 * np.pi * np.arange(n) * rlf
        cols = [np.ones(n), np.cos(wt), np.sin(wt),
                np.cos(wt / 2), np.sin(wt / 2),
                np.cos(wlf), np.sin(wlf), np.cos(2*wlf), np.sin(2*wlf)]
        for h in range(2, Nhars + 1):
            cols.extend([np.cos(h * wt), np.sin(h * wt)])
        # If any line harmonic falls within fline/2 of fsig, add it explicitly so it
        # doesn't leak into the signal estimate (e.g. 42×60=2520 Hz near 2512 Hz).
        n_extra_line = 0
        n_nearest = int(round(fsig / fline))
        for nh in range(max(3, n_nearest - 2), n_nearest + 3):
            fn = nh * fline
            if fsamp / n < abs(fn - fsig) < fline / 2:
                wnh = 2 * np.pi * np.arange(n) * fn / fsamp
                cols.extend([np.cos(wnh), np.sin(wnh)])
                n_extra_line += 2
        X = np.column_stack(cols)
        w = np.hanning(n) if use_hann else np.ones(n)
        X_eff = X * w[:, np.newaxis]
        y_eff = y * w
        fit_pars, lstsq_res, _, _ = np.linalg.lstsq(X_eff, y_eff, rcond=None)
        chunk_size = int(round(chunk_periods * fsamp / fsig)) if chunk_periods > 0 else 0
        use_chunks = chunk_size > n_chunk_params + 1 and n // chunk_size >= 2

    fit_vals = X @ fit_pars
    rss  = float(np.sum((w * (y - fit_vals)) ** 2))
    ndf  = n - X.shape[1]
    errv = np.sqrt(rss / ndf) if ndf > 0 else 0.0

    if not use_chunks:
        return fit_pars[1] - 1j * fit_pars[2], fit_vals, errv, rss

    # Stage 1: remove background (DC, line harmonics, f/2) leaving signal only
    bg_pars    = fit_pars.copy()
    bg_pars[1] = 0.0
    bg_pars[2] = 0.0
    y_bg  = X @ bg_pars
    y_sig = y - y_bg

    # Stage 2: per-chunk amplitudes
    if cache is not None:
        n_chunks    = cache['n_chunks']
        chunk_size  = cache['chunk_size']
        chunk_pinvs = cache['chunk_pinvs']
    else:
        n_chunks    = n // chunk_size
        chunk_pinvs = None

    chunk_amps = np.zeros(n_chunks, dtype=complex)
    fit_sig    = np.zeros(n, dtype=float)
    rss_total  = 0.0
    ndf_total  = 0

    for k in range(n_chunks):
        i0 = k * chunk_size
        i1 = i0 + chunk_size
        yc = y_sig[i0:i1]
        if chunk_pinvs is not None:
            pinv_c, wc, Xc = chunk_pinvs[k]
            pars_c = pinv_c @ (yc * wc)
        else:
            idx = np.arange(i0, i1)
            wtc = 2 * np.pi * idx * rf
            Xc  = np.column_stack([np.ones(chunk_size),
                                    np.cos(wtc), np.sin(wtc),
                                    np.cos(wtc / 2), np.sin(wtc / 2)])
            wc = np.hanning(chunk_size) if use_hann else np.ones(chunk_size)
            pars_c, _, _, _ = np.linalg.lstsq(Xc * wc[:, np.newaxis], yc * wc, rcond=None)
        chunk_amps[k] = pars_c[1] - 1j * pars_c[2]
        fit_sig[i0:i1] = Xc @ pars_c
        rss_total += float(np.sum((yc - Xc @ pars_c) ** 2))
        ndf_total += chunk_size - n_chunk_params

    # Tail samples not covered by whole chunks: use global fundamental
    tail = n_chunks * chunk_size
    if tail < n:
        tail_idx = np.arange(tail, n)
        fit_sig[tail:] = (fit_pars[1] * np.cos(2 * np.pi * tail_idx * rf) +
                          fit_pars[2] * np.sin(2 * np.pi * tail_idx * rf))

    complex_amp = np.mean(chunk_amps)
    fit_vals    = y_bg + fit_sig
    errv        = np.sqrt(rss_total / ndf_total) if ndf_total > 0 else 0.0
    return complex_amp, fit_vals, errv, rss_total, chunk_amps


def get_f(y, fsamp, fsig_guess, fline_guess=60.0, use_hann=True, Nhars=1, n_coarse=21):
    """
    Estimates the signal frequency by minimizing the residual sum of squares
    from fit_sine_cplx. A coarse grid scan over the search bracket locates
    the right neighborhood first, then Brent's bounded method polishes
    within one grid spacing of the coarse minimum for high precision.

    The coarse step exists because scipy.optimize.minimize_scalar's bounded
    Brent search assumes a roughly unimodal objective; with real noisy data
    the RSS-vs-frequency curve can have local wiggles that trap a bare Brent
    search in the wrong minimum, letting a small residual frequency error
    slip through undetected.

    chunk_periods is intentionally left at 0 here for speed.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    # Search +/- one full FFT bin around the guess. The old +/-fsig/n bracket
    # (a few mHz wide at typical n) was tighter than the spectral resolution
    # itself, so Brent's method would pin at the edge whenever the true
    # frequency drifted from fsig_guess by more than that, leaving an
    # un-cancelled spike in the residual right at the true frequency.
    fbin = fsamp / n
    fsig_min = fsig_guess - fbin
    fsig_max = fsig_guess + fbin

    def rss_at(fsig):
        return fit_sine_cplx(y, fsamp, fsig, fline_guess, Nhars=Nhars, use_hann=use_hann)[3]

    grid = np.linspace(fsig_min, fsig_max, n_coarse)
    rss_grid = np.array([rss_at(f) for f in grid])
    i_best = np.argmin(rss_grid)
    grid_spacing = grid[1] - grid[0]
    polish_min = max(fsig_min, grid[i_best] - grid_spacing)
    polish_max = min(fsig_max, grid[i_best] + grid_spacing)

    res_sig = scipy.optimize.minimize_scalar(
        rss_at,
        bounds=(polish_min, polish_max),
        method='bounded',
    )
    best_fsig = res_sig.x

    res_line = scipy.optimize.minimize_scalar(
        lambda fline: fit_sine_cplx(y, fsamp, best_fsig, fline, Nhars=Nhars, use_hann=use_hann)[3],
        bounds=(fline_guess - 0.5, fline_guess + 0.5),
        method='bounded',
    )
    return best_fsig, res_line.x


def fit_eta_regression(eta1, eta3):
    """Complex linear regression eta1 = mgain1*eta3 + const, returning
    (mgain1, mratio1), or (None, None) if there aren't enough points yet.

    eta1 = V1'/V2' and eta3 = V3'/V2' (both normalized by the Z_REF-branch
    voltage V2', per the bridge derivation). Solving the circuit equations
    for V3' gives Yref*Z_DUT = -(Z_DUT/Z_FB)*eta3 - eta1 exactly, at every
    instant -- where Yref = j*omega*C for a capacitive reference or 1/R for
    a resistive one -- and Z_DUT/Z_FB and Yref*Z_DUT are both physical
    constants that don't move with the modulation, so eta1 is an exact
    linear function of eta3 with complex slope mgain1 = -(Z_DUT/Z_FB),
    regardless of the modulation trajectory's shape. A direct regression is
    therefore less restrictive than fitting an ellipse to each and taking
    an axis-ratio gain. The fitted intercept mratio1 = mgain1*eta3_mean -
    eta1_mean equals Yref*Z_DUT directly, so Z_DUT = mratio1/Yref.
    Two complex points are the minimum needed to determine the two complex
    unknowns (slope and offset)."""
    eta1 = np.asarray(eta1)
    eta3 = np.asarray(eta3)
    if len(eta1) < 2:
        return None, None
    eta1_mean = np.mean(eta1)
    eta3_mean = np.mean(eta3)
    d_eta1 = eta1 - eta1_mean
    d_eta3 = eta3 - eta3_mean
    denom = np.sum(np.abs(d_eta3)**2)
    if denom == 0:
        return None, None
    mgain1  = np.dot(d_eta1, np.conj(d_eta3)) / denom
    mratio1 = mgain1*eta3_mean - eta1_mean
    return mgain1, mratio1


def build_fit_cache(fsamp, fsig, fline, n, Nhars, use_hann=True, chunk_periods=0):
    """Precompute fit matrices for repeated calls at the same frequency and sample count.

    Builds the full design matrix X, the Hanning window w, and the pseudoinverse
    pinv(X * w[:, None]).  When chunk_periods > 0, also precomputes the per-chunk
    pseudoinverses so Stage-2 fits reduce to matrix–vector multiplies.

    Pass the returned dict as cache= to fit_sine_cplx.
    """
    rf  = fsig  / fsamp
    rlf = fline / fsamp
    wt  = 2 * np.pi * np.arange(n) * rf
    wlf = 2 * np.pi * np.arange(n) * rlf

    cols = [np.ones(n), np.cos(wt), np.sin(wt),
            np.cos(wt / 2), np.sin(wt / 2),
            np.cos(wlf), np.sin(wlf), np.cos(2*wlf), np.sin(2*wlf)]
    for h in range(2, Nhars + 1):
        cols.extend([np.cos(h * wt), np.sin(h * wt)])

    n_nearest = int(round(fsig / fline))
    for nh in range(max(3, n_nearest - 2), n_nearest + 3):
        fn = nh * fline
        if fsamp / n < abs(fn - fsig) < fline / 2:
            wnh = 2 * np.pi * np.arange(n) * fn / fsamp
            cols.extend([np.cos(wnh), np.sin(wnh)])

    X = np.column_stack(cols)
    w = np.hanning(n) if use_hann else np.ones(n)
    bg_pinv = np.linalg.pinv(X * w[:, np.newaxis])

    cache = {'X': X, 'w': w, 'bg_pinv': bg_pinv, 'rf': rf}

    n_chunk_params = 5
    chunk_size = int(round(chunk_periods * fsamp / fsig)) if chunk_periods > 0 else 0
    if chunk_size > n_chunk_params + 1 and n // chunk_size >= 2:
        n_chunks    = n // chunk_size
        chunk_pinvs = []
        for k in range(n_chunks):
            i0  = k * chunk_size
            i1  = i0 + chunk_size
            idx = np.arange(i0, i1)
            wtc = 2 * np.pi * idx * rf
            Xc  = np.column_stack([np.ones(chunk_size),
                                    np.cos(wtc), np.sin(wtc),
                                    np.cos(wtc / 2), np.sin(wtc / 2)])
            wc = np.hanning(chunk_size) if use_hann else np.ones(chunk_size)
            chunk_pinvs.append((np.linalg.pinv(Xc * wc[:, np.newaxis]), wc, Xc))
        cache['chunk_pinvs'] = chunk_pinvs
        cache['chunk_size']  = chunk_size
        cache['n_chunks']    = n_chunks

    return cache


@dataclass
class ComplexEllipse:
    """
    Represents an ellipse parameterized by counter-rotating complex amplitudes.
    Optimized for R2F converter signal processing.
    """
    eta_o: complex
    eta_ccw: complex
    eta_cw: complex

    @property
    def center(self) -> tuple[float, float]:
        """Returns the geometric center (cx, cy)."""
        return np.real(self.eta_o), np.imag(self.eta_o)

    @property
    def semi_major(self) -> float:
        """Returns the semi-major axis length."""
        return complex(self.eta_ccw + self.eta_cw)

    @property
    def semi_minor(self) -> float:
        """Returns the semi-minor axis length."""
        return complex((self.eta_ccw - self.eta_cw)*np.exp(1j*np.pi/2))

    @property
    def angle(self) -> float:
        """Returns the rotation angle in radians."""
        return float(np.angle(self.eta_ccw + self.eta_cw))

    def evaluate(self, N: int = 100) -> np.ndarray:
        """Generates the complex trajectory for N points."""
        n = np.arange(N)
        phase = 2 * np.pi * n / N
        return self.eta_o + self.eta_ccw * np.exp(1j * phase) + self.eta_cw * np.exp(-1j * phase)

    @classmethod
    def fit_from_points(cls, x: np.ndarray, y: np.ndarray):
        """
        Fits an ellipse to (x,y) data using the Fitzgibbon Direct Least Squares method
        and returns a pure ComplexEllipse instance.
        """


        x = np.array(x)
        y = np.array(y)

        mx, my = np.mean(x), np.mean(y)
        scale = np.max([np.std(x), np.std(y)])
        if scale == 0:
            return None

        xn = (x - mx) / scale
        yn = (y - my) / scale

        # 1. Construct matrices
        D = np.vstack([xn**2, xn*yn, yn**2, xn, yn, np.ones_like(xn)]).T
        S = np.dot(D.T, D)

        C = np.zeros((6, 6))
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1

        # 2. Solve the eigensystem
        try:
            epsilon = 1e-10  # Regularization for noiseless edge cases
            S = S + epsilon * np.eye(6)
            E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
        except np.linalg.LinAlgError:
            return None

        # 3. Extract algebraic coefficients
        n = np.argmax(E)
        A, B, C_coeff, D_coeff, E_coeff, F = V[:, n]

        # 4. Convert Algebraic to Geometric Parameters
        # Center coordinates
        denominator = B**2 - 4 * A * C_coeff
        if denominator >= 0:
            return None # Not an ellipse

        cx = (2 * C_coeff * D_coeff - B * E_coeff) / denominator
        cy = (2 * A * E_coeff - B * D_coeff) / denominator

        # Semi-axes lengths
        numerator = 2 * (A * E_coeff**2 + C_coeff * D_coeff**2 - B * D_coeff * E_coeff + denominator * F)
        term2 = np.sqrt((A - C_coeff)**2 + B**2)

        axis1 = np.sqrt(abs(numerator / (denominator * (A + C_coeff + term2))))
        axis2 = np.sqrt(abs(numerator / (denominator * (A + C_coeff - term2))))

        major = max(axis1, axis2)
        minor = min(axis1, axis2)
        # Rotation angle
        theta = 0.5 * np.arctan2(B, A - C_coeff)
        phi = np.arctan2(y-cy,x-cx)
        x_unrotated = major * np.cos(phi)
        y_unrotated = minor * np.sin(phi)
        test1_x = x_unrotated * np.cos(theta) - y_unrotated * np.sin(theta) + cx
        test1_y = x_unrotated * np.sin(theta) + y_unrotated * np.cos(theta) + cy

        residual1  = (A * test1_x**2 + B * test1_x * test1_y + C_coeff * test1_y**2 +
                             D_coeff * test1_x + E_coeff * test1_y + F)
        res1= np.dot(residual1 ,residual1 )
        test2_x = x_unrotated * np.cos(theta+np.pi/2) - y_unrotated * np.sin(theta+np.pi/2) + cx
        test2_y = x_unrotated * np.sin(theta+np.pi/2) + y_unrotated * np.cos(theta+np.pi/2) + cy

        residual2  = (A * test2_x**2 + B * test2_x * test2_y + C_coeff * test2_y**2 +
                             D_coeff * test2_x + E_coeff * test2_y + F)
        res2= np.dot(residual2 ,residual2 )

        if res2<res1:
            theta+=np.pi/2
            c2=res2
        else:
            c2=res1

        phi0 = np.arctan2(yn[0]-cy,xn[0]-cx)
        #print(phi0,theta)


        # 5. Calculate Counter-Rotating Complex Amplitudes
        eta_o = cx*scale+mx + 1j *(( cy*scale) +my)
        #phase_factor = np.exp(1j * theta)

        a = np.exp(1j*phi0)
        b = np.exp(1j*theta)
        c = np.exp(1j*(theta+np.pi))
        if abs(np.angle(c/a))<np.abs(np.angle(b/a)):
            theta = theta+np.pi


        #phase_factor = np.exp(1j * phi0)
        phase_factor = np.exp(1j * theta)
        eta_ccw = 0.5 * (major + minor) * phase_factor *scale
        eta_cw = 0.5 * (major - minor) * phase_factor *scale

        return cls(eta_o=eta_o, eta_ccw=eta_ccw, eta_cw=eta_cw)

    @classmethod
    def fit_from_cmplx_points(cls, cplx):
        return cls.fit_from_points(np.real(cplx),np.imag(cplx))

    def simulate_noisy_data(self, N: int, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates N points along the ellipse with optional Gaussian noise.
        Returns a tuple of (x_array, y_array) to be used for testing.
        """
        # 1. Get the perfect mathematical path
        z_perfect = self.evaluate(N)

        # 2. Add random Gaussian noise to the real and imaginary parts
        x = np.real(z_perfect) + np.random.normal(0, noise_std, N)
        y = np.imag(z_perfect) + np.random.normal(0, noise_std, N)

        return x, y

    def plot_elli(self,ax,ellipse_color='k',maj_color='r',min_color='b',rescale=True):
        data = self.evaluate()
        ax.plot(np.real(data),np.imag(data),linestyle='-',color=ellipse_color)
        l1 = np.array([self.eta_o,self.eta_o+self.semi_major])
        l2 = np.array([self.eta_o,self.eta_o+self.semi_minor])
        ax.plot(np.real(l1),np.imag(l1),linestyle='-.',color= maj_color)
        ax.plot(np.real(l2),np.imag(l2),linestyle=':',color=min_color)
        if rescale:
            bex,enx = ax.get_xlim()
            bey,eny = ax.get_ylim()
            delx,mex = 0.5*(enx-bex), 0.5*(enx+bex)
            dely,mey = 0.5*(eny-bey), 0.5*(eny+bey)
            if delx>dely:
                delm=delx
            else:
                delm=dely

            bex,enx = mex-delm ,mex+delm
            bey,eny = mey-delm ,mey+delm
            ax.set_xlim(bex,enx)
            ax.set_ylim(bey,eny)
