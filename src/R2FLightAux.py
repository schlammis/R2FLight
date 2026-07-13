import numpy as np
from dataclasses import dataclass

"""
$\hat{\chi}^2 = a +bx +cx^2$
Minimum:
$ x =-\frac{b}{2c}$ has the value $a-\frac{b^2}{2c} + \frac{b^2}{4c} = a-\frac{b^2}{4c}$
At the Minimum the $\chi^2$ should be the $NDF=N-3$. So,
$\chi^2 =\frac{\hat{\chi}^2}{\sigma^2}=N-3$. Hence, $\sigma^2 = \frac{\hat{\chi}^2}{N-3}$
The uncertainty in the parameter is  when the unscaled $\chi^2$ goes up by 1. That means the scaled one goes up by $\sigma^2$
$cx^2 = \sigma^2$ => $x=\sigma/c$
"""

def fit_sine(y,rf):
    """
    rf = relative frequency = f0/fs.
    sin(w t)= sin (2*pi*f0 *i/fs)= sin(2*pi*i *f0/fs)
    A sin(wt + phi) = A*sin(wt)*cos(phi)+A*cos(wt)*sin(phi)=S*sin(wt)+C*cos(w) => C/S=A*sin(phi)/A*cos(phi)=tan(phi) => phi=atan(C/S)

    If we write
    y  = Re( A*Exp(i (wt +phi)) = Re( (A* cos( w+ phi) + A*i*sin(w + phi))
    Re(A)* cos(wt + phi) - Im(A)* sin(w*t+phi)
    Re => cos_coeff
    Im => -sin_coeff

    """
    i = np.arange(len(y))
    wt= 2*np.pi*i*rf
    O = np.ones(len(y))
    C = np.cos(wt)
    S = np.sin(wt)
    X = np.vstack((O,C,S)).T
    fit_pars = np.linalg.solve(X.T @ X, X.T @ y)
    fit_vals = X @ fit_pars
    C2 = np.dot(y-fit_vals,y-fit_vals)
    return fit_pars[1],fit_pars[2],fit_vals,C2

def fit_sine_cplx(y,rf,useHann=True):
    """
    rf = relative frequency = f0/fs.
    sin(w t)= sin (2*pi*f0 *i/fs)= sin(2*pi*i *f0/fs)
    A sin(wt + phi) = A*sin(wt)*cos(phi)+A*cos(wt)*sin(phi)=S*sin(wt)+C*cos(w) => C/S=A*sin(phi)/A*cos(phi)=tan(phi) => phi=atan(C/S)

    If we write
    y  = Re( A*Exp(i (wt +phi)) = Re( (A* cos( w+ phi) + A*i*sin(w + phi))
    Re(A)* cos(wt + phi) - Im(A)* sin(w*t+phi)
    Re => cos_coeff
    Im => -sin_coeff

    """
    i = np.arange(len(y))
    wt= 2*np.pi*i*rf
    O = np.ones(len(y))
    C = np.cos(wt)
    S = np.sin(wt)
    X = np.vstack((O,C,S)).T
    if useHann:
        #W_vec = np.blackman(len(y))
        W_vec = np.hanning(len(y))
        yw = y * W_vec
        Ow = O * W_vec
        Cw = C * W_vec
        Sw = S * W_vec
        Xw = np.vstack((Ow,Cw,Sw)).T
        fit_pars = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)
    else:
        fit_pars = np.linalg.solve(X.T @ X, X.T @ y)
    fit_vals = X @ fit_pars
    C2 = np.dot(y-fit_vals,y-fit_vals)
    NDF = len(y)-3
    errv = np.sqrt(C2/NDF)
    return fit_pars[1]-1j*fit_pars[2],fit_vals,errv

def get_f(y,rf):
    """
    y = a+b*f+cf^2 => min b +2*c*f=0 => f= -b/(2c)
    """
    N = len(y)
    _,_,_,C20 = fit_sine(y,rf)
    _,_,_,C2m = fit_sine(y,rf*N/(N+1))
    _,_,_,C2p = fit_sine(y,rf*N/(N-1))
    ff = [rf*N/(N+1),rf,rf*N/(N-1)]
    yy = [C2m,C20,C2p]
    pf = np.polyfit(ff,yy,2)
    minf = -pf[1]/2/pf[0]
    if minf>rf*N/(N+1) and minf<rf*N/(N-1):
        return minf
    else:
        return get_f(y,minf)

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
