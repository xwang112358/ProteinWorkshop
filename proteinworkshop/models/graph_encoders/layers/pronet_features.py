"""Geometric feature encodings for ProNet using spherical harmonics and Bessel functions.

This module ports the DIVE implementation of ProNet's geometric encoding utilities.
"""
import torch
import torch.nn as nn
import sympy as sym
import numpy as np
from scipy.optimize import brentq
from scipy import special as sp


def Jn(r, n):
    """Numerical spherical Bessel functions of order n.
    
    Args:
        r: Radial distance
        n: Order of the Bessel function
        
    Returns:
        Value of the spherical Bessel function of order n at r
    """
    return sp.spherical_jn(n, r)


def Jn_zeros(n, k):
    """Compute the first k zeros of the spherical Bessel functions up to order n (excluded).
    
    Args:
        n: Maximum order (excluded)
        k: Number of zeros to compute
        
    Returns:
        Array of shape (n, k) containing the zeros
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    """Computes the sympy formulas for the spherical Bessel functions up to order n (excluded).
    
    Uses the formula: j_i = (-x)^i * (1/x * d/dx)^i * sin(x)/x
    
    Args:
        n: Maximum order (excluded)
        
    Returns:
        List of sympy expressions for the spherical Bessel functions
    """
    x = sym.symbols("x")
    j = [sym.sin(x) / x]  # j_0
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        j += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return j


def bessel_basis(n, k):
    """Compute the sympy formulas for the normalized and rescaled spherical Bessel functions.
    
    Computes formulas up to order n (excluded) and maximum frequency k (excluded).
    
    Args:
        n: Maximum order (excluded)
        k: Maximum frequency (excluded)
        
    Returns:
        List of length n, where each element is a list of length k containing
        normalized Bessel basis formulas. In total n*k formulas.
    """
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = (
            1 / np.array(normalizer_tmp) ** 0.5
        )  # sqrt(2/(j_l+1)**2), sqrt(1/c**3) not taken into account yet
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols("x")
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(
                    normalizer[order][i] * f[order].subs(x, zeros[order, i] * x)
                )
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l, m):
    """Computes the constant pre-factor for the spherical harmonic of degree l and order m.
    
    Formula: sqrt((2*l+1)/4*pi * (l-m)!/(l+m)!)
    
    Args:
        l: Degree of the spherical harmonic (l >= 0)
        m: Order of the spherical harmonic (-l <= m <= l)
        
    Returns:
        Normalization factor
    """
    return (
        (2 * l + 1)
        / (4 * np.pi)
        * np.math.factorial(l - abs(m))
        / np.math.factorial(l + abs(m))
    ) ** 0.5


def associated_legendre_polynomials(L, zero_m_only=True, pos_m_only=True):
    """Computes string formulas of the associated Legendre polynomials up to degree L (excluded).
    
    Args:
        L: Degree up to which to calculate (degree L is excluded)
        zero_m_only: If True, only calculate polynomials where m=0
        pos_m_only: If True, only calculate polynomials where m>=0 (overwritten by zero_m_only)
        
    Returns:
        List of lists containing sympy formulas for the polynomials
        (L many if zero_m_only is True, else L^2 many)
    """
    z = sym.symbols("z")
    P_l_m = [[0] * (2 * l + 1) for l in range(L)]  # for order l: -l <= m <= l

    P_l_m[0][0] = 1
    if L > 0:
        if zero_m_only:
            # m = 0
            P_l_m[1][0] = z
            for l in range(2, L):
                P_l_m[l][0] = sym.simplify(
                    ((2 * l - 1) * z * P_l_m[l - 1][0] - (l - 1) * P_l_m[l - 2][0]) / l
                )
            return P_l_m
        else:
            # for m >= 0
            for l in range(1, L):
                P_l_m[l][l] = sym.simplify(
                    (1 - 2 * l) * (1 - z ** 2) ** 0.5 * P_l_m[l - 1][l - 1]
                )

            for m in range(0, L - 1):
                P_l_m[m + 1][m] = sym.simplify(
                    (2 * m + 1) * z * P_l_m[m][m]
                )

            for l in range(2, L):
                for m in range(l - 1):
                    P_l_m[l][m] = sym.simplify(
                        (
                            (2 * l - 1) * z * P_l_m[l - 1][m]
                            - (l + m - 1) * P_l_m[l - 2][m]
                        )
                        / (l - m)
                    )

            if not pos_m_only:
                # for m < 0: P_l(-m) = (-1)^m * (l-m)!/(l+m)! * P_lm
                for l in range(1, L):
                    for m in range(1, l + 1):
                        P_l_m[l][-m] = sym.simplify(
                            (-1) ** m
                            * np.math.factorial(l - m)
                            / np.math.factorial(l + m)
                            * P_l_m[l][m]
                        )

            return P_l_m


def real_sph_harm(L, spherical_coordinates=True, zero_m_only=True):
    """Computes formula strings of the real part of the spherical harmonics up to degree L (excluded).
    
    Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z)
    on the UNIT SPHERE.
    
    Args:
        L: Degree up to which to calculate (degree L is excluded)
        spherical_coordinates: If True, expects input as (phi, theta). If False, expects (x, y, z)
        zero_m_only: If True, only calculate harmonics where m=0
        
    Returns:
        List of lists containing sympy formulas for the real spherical harmonics.
        L^2 harmonics in total up to degree L (or L many if zero_m_only is True)
    """
    z = sym.symbols("z")
    P_l_m = associated_legendre_polynomials(L, zero_m_only)
    if zero_m_only:
        Y_l_m = [[0] for l in range(L)]
    else:
        Y_l_m = [[0] * (2 * l + 1) for l in range(L)]

    # Convert expressions to spherical coordinates
    if spherical_coordinates:
        theta = sym.symbols("theta")
        for l in range(L):
            for m in range(len(P_l_m[l])):
                if not isinstance(P_l_m[l][m], int):
                    P_l_m[l][m] = P_l_m[l][m].subs(z, sym.cos(theta))

    # Calculate Y_lm
    # Y_lm = N * P_lm(cos(theta)) * exp(i*m*phi)
    # Real form depends on sign of m
    for l in range(L):
        Y_l_m[l][0] = sym.simplify(sph_harm_prefactor(l, 0) * P_l_m[l][0])

    if not zero_m_only:
        phi = sym.symbols("phi")
        for l in range(1, L):
            # m > 0
            for m in range(1, l + 1):
                Y_l_m[l][m] = sym.simplify(
                    2 ** 0.5
                    * (-1) ** m
                    * sph_harm_prefactor(l, m)
                    * P_l_m[l][m]
                    * sym.cos(m * phi)
                )
            # m < 0
            for m in range(1, l + 1):
                Y_l_m[l][-m] = sym.simplify(
                    2 ** 0.5
                    * (-1) ** m
                    * sph_harm_prefactor(l, -m)
                    * P_l_m[l][m]
                    * sym.sin(m * phi)
                )

        # Convert expressions to cartesian coordinates
        if not spherical_coordinates:
            x = sym.symbols("x")
            y = sym.symbols("y")
            for l in range(L):
                for m in range(len(Y_l_m[l])):
                    Y_l_m[l][m] = sym.simplify(Y_l_m[l][m].subs(phi, sym.atan2(y, x)))
    return Y_l_m


class AngleEmbedding(nn.Module):
    """Embed (distance, angle) pairs using Bessel functions × Spherical harmonics.
    
    Used for encoding Euler angles in backbone representation.
    Pre-computes basis functions during initialization for efficient evaluation.
    
    Args:
        num_radial: Number of radial basis functions
        num_spherical: Number of spherical harmonic degrees
        cutoff: Cutoff distance for normalization (default: 8.0 Angstroms)
    """
    
    def __init__(self, num_radial: int, num_spherical: int, cutoff: float = 8.0):
        super().__init__()
        assert num_radial <= 64, "num_radial must be <= 64"
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff

        # Compute symbolic basis functions
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=True
        )
        
        # Convert to lambda functions for efficient evaluation
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols("x")
        theta = sym.symbols("theta")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        
        m = 0
        for l in range(len(Y_lm)):
            if l == 0:
                # Special handling for l=0 (constant function)
                first_sph = sym.lambdify([theta], Y_lm[l][m], modules)
                self.sph_funcs.append(
                    lambda theta: torch.zeros_like(theta) + first_sph(theta)
                )
            else:
                self.sph_funcs.append(sym.lambdify([theta], Y_lm[l][m], modules))
            for n in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][n], modules)
                )

    def forward(self, dist: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Compute angle embedding.
        
        Args:
            dist: Edge distances (num_edges,)
            angle: Angles in radians (num_edges,)
            
        Returns:
            Embedded features (num_edges, num_spherical * num_radial)
        """
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        sbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)
        n, k = self.num_spherical, self.num_radial
        out = (rbf.view(-1, n, k) * sbf.view(-1, n, 1)).view(-1, n * k)
        return out


class ThetaPhiEmbedding(nn.Module):
    """Embed (distance, theta, phi) triplets using Bessel functions × Spherical harmonics.
    
    Used for encoding angular geometry in complete 3D graph.
    Computes full spherical harmonics (not just m=0).
    
    Args:
        num_radial: Number of radial basis functions
        num_spherical: Number of spherical harmonic degrees
        cutoff: Cutoff distance for normalization (default: 8.0 Angstroms)
    """
    
    def __init__(self, num_radial: int, num_spherical: int, cutoff: float = 8.0):
        super().__init__()
        assert num_radial <= 64, "num_radial must be <= 64"
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff

        # Compute symbolic basis functions
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=False
        )
        
        # Convert to lambda functions for efficient evaluation
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols("x")
        theta = sym.symbols("theta")
        phi = sym.symbols("phi")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        
        for l in range(len(Y_lm)):
            for m in range(len(Y_lm[l])):
                if l == 0:
                    # Special handling for l=0 (constant function)
                    first_sph = sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    self.sph_funcs.append(
                        lambda theta, phi: torch.zeros_like(theta)
                                           + first_sph(theta, phi)
                    )
                else:
                    self.sph_funcs.append(
                        sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    )
            for j in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][j], modules)
                )

        self.register_buffer(
            "degreeInOrder", torch.arange(num_spherical) * 2 + 1, persistent=False
        )

    def forward(
        self, dist: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        """Compute theta-phi embedding.
        
        Args:
            dist: Edge distances (num_edges,)
            theta: Polar angles in radians (num_edges,)
            phi: Azimuthal angles in radians (num_edges,)
            
        Returns:
            Embedded features (num_edges, num_spherical^2 * num_radial)
        """
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        sbf = torch.stack([f(theta, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        rbf = (
            rbf.view((-1, n, k))
            .repeat_interleave(self.degreeInOrder, dim=1)
            .view((-1, n ** 2 * k))
        )
        sbf = sbf.repeat_interleave(k, dim=1)
        out = rbf * sbf
        return out
