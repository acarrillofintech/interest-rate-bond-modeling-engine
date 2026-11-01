# Mathematical Methodology

## Interest Rate and Bond Modeling Engine

**Author:** Alexander Carrillo Reyes  
**Version:** 1.0.0  
**Language:** Python

---

## 1. Purpose

This document describes the mathematical foundations, numerical methods, assumptions, and implementation decisions used by the Interest Rate and Bond Modeling Engine.

The engine provides tools for:

- Fixed-coupon and zero-coupon bond pricing.
- Cash-flow generation.
- Current yield and Yield to Maturity.
- Effective annual yield.
- Macaulay and modified duration.
- Convexity and DV01.
- Spot rates and discount factors.
- Forward-rate calculations.
- Yield-curve interpolation.
- Zero-curve bootstrapping.
- Vasicek short-rate simulation.
- Cox-Ingersoll-Ross short-rate simulation.
- Monte Carlo scenario analysis.
- Fixed-income visualizations.

---

## 2. Financial Conventions

The project uses the following conventions:

- Rates are expressed as decimals.
- Cash flows occur at regular intervals.
- Coupon rates are nominal annual rates.
- Yields are nominal annual rates compounded according to the bond's payment frequency.
- The base implementation assumes deterministic bond cash flows.
- Default risk, liquidity risk, taxes, and transaction costs are excluded.
- A basis point is equal to:

\[
1\text{ bp}=0.0001=0.01\%
\]

The number of coupon periods is:

\[
N=T\times m
\]

where:

- \(T\) is the maturity in years.
- \(m\) is the number of payments per year.
- \(N\) must be an integer.

---

## 3. Bond Cash Flows

For a fixed-rate bond, the periodic coupon is:

\[
C=\frac{F\times c}{m}
\]

where:

- \(F\) is the face value.
- \(c\) is the annual coupon rate.
- \(m\) is the number of coupon payments per year.

For all periods before maturity:

\[
CF_t=C
\]

At maturity:

\[
CF_N=C+F
\]

A five-year bond with semiannual payments has:

\[
N=5\times2=10
\]

coupon periods.

---

## 4. Fixed-Coupon Bond Pricing

The price of a bond is the present value of all future contractual cash flows:

\[
P=
\sum_{t=1}^{N}
\frac{C}{
\left(1+\frac{y}{m}\right)^t
}
+
\frac{F}{
\left(1+\frac{y}{m}\right)^N
}
\]

where:

- \(P\) is the bond price.
- \(C\) is the periodic coupon.
- \(F\) is the face value.
- \(y\) is the nominal annual yield.
- \(m\) is the payment frequency.
- \(N\) is the number of payment periods.

Equivalently:

\[
P=
\sum_{t=1}^{N}
\frac{CF_t}{
\left(1+\frac{y}{m}\right)^t
}
\]

### Pricing Interpretation

If:

\[
c>y
\]

the bond trades above face value and is described as a premium bond.

If:

\[
c=y
\]

the bond trades at face value.

If:

\[
c<y
\]

the bond trades below face value and is described as a discount bond.

Bond prices and market yields move in opposite directions.

---

## 5. Zero-Coupon Bond Pricing

A zero-coupon bond makes no periodic coupon payments. Its only cash flow is the face value paid at maturity.

Its price is:

\[
P_0=
\frac{F}{
\left(1+\frac{y}{m}\right)^{mT}
}
\]

The difference between the face value and the purchase price represents the investor's return.

---

## 6. Current Yield

Current yield compares annual coupon income with market price:

\[
\text{Current Yield}
=
\frac{F\times c}{P}
\]

Current yield does not incorporate:

- The time value of money.
- Capital gains or losses at maturity.
- Reinvestment of coupons.
- The remaining maturity of the bond.

Therefore, it is a partial yield measure rather than a complete return measure.

---

## 7. Yield to Maturity

Yield to Maturity is the discount rate that equates the present value of the bond's future cash flows with its observed market price:

\[
P_{\text{market}}
=
\sum_{t=1}^{N}
\frac{CF_t}{
\left(1+\frac{YTM}{m}\right)^t
}
\]

The YTM generally cannot be isolated algebraically for a coupon bond. It is obtained by solving:

\[
f(y)=
P(y)-P_{\text{market}}=0
\]

The implementation uses Brent's root-finding method.

### YTM Assumptions

The conventional YTM interpretation assumes:

- The bond is held until maturity.
- All contractual payments are received.
- Coupons can be reinvested at the YTM.
- The issuer does not default.
- There are no transaction costs or taxes.

The numerical domain requires:

\[
y>-m
\]

so that the periodic discount factor remains positive.

---

## 8. Effective Annual Yield

A nominal annual yield compounded \(m\) times per year is converted into an effective annual yield using:

\[
EAY=
\left(1+\frac{y}{m}\right)^m-1
\]

The reverse conversion is:

\[
y=
m\left[
(1+EAY)^{1/m}-1
\right]
\]

The effective annual yield captures the effect of intra-year compounding.

---

## 9. Macaulay Duration

Macaulay duration is the present-value-weighted average time required to receive the bond's cash flows:

\[
D_{\text{Mac}}
=
\frac{
\sum_{t=1}^{N}
T_t\times PV(CF_t)
}{
P
}
\]

where:

\[
T_t=\frac{t}{m}
\]

and:

\[
PV(CF_t)=
\frac{CF_t}{
\left(1+\frac{y}{m}\right)^t
}
\]

Macaulay duration is measured in years.

For a zero-coupon bond:

\[
D_{\text{Mac}}=T
\]

because the complete value is received at maturity.

---

## 10. Modified Duration

Modified duration converts Macaulay duration into a first-order measure of price sensitivity:

\[
D_{\text{Mod}}
=
\frac{
D_{\text{Mac}}
}{
1+\frac{y}{m}
}
\]

For a small yield movement:

\[
\frac{\Delta P}{P}
\approx
-D_{\text{Mod}}\Delta y
\]

Therefore:

\[
\Delta P
\approx
-PD_{\text{Mod}}\Delta y
\]

The negative sign represents the inverse relationship between bond prices and yields.

Duration is a linear approximation and becomes less precise for large interest-rate movements.

---

## 11. Convexity

The bond price-yield relationship is curved rather than linear. Convexity measures this curvature.

For periodic compounding:

\[
\text{Convexity}
=
\frac{1}{Pm^2}
\sum_{t=1}^{N}
\frac{
CF_t\times t(t+1)
}{
\left(1+\frac{y}{m}\right)^{t+2}
}
\]

The duration-and-convexity approximation is:

\[
\frac{\Delta P}{P}
\approx
-D_{\text{Mod}}\Delta y
+
\frac{1}{2}
\text{Convexity}
(\Delta y)^2
\]

Convexity improves the approximation because it incorporates the second derivative of price with respect to yield.

For a standard option-free bond, convexity is generally positive.

---

## 12. DV01

DV01 measures the approximate monetary change in bond price caused by a one-basis-point movement in yield.

\[
DV01=
P\times D_{\text{Mod}}\times0.0001
\]

For example, a DV01 of USD 0.45 indicates that a one-basis-point yield movement changes the bond's value by approximately USD 0.45 in the opposite direction.

DV01 is also referred to as:

- Dollar value of one basis point.
- Basis-point value.
- PV01 in some fixed-income contexts.

---

## 13. Discount Factors

A discount factor represents the present value of one monetary unit received at a future maturity.

For a nominal spot rate \(s_T\):

\[
DF(0,T)
=
\frac{1}{
\left(1+\frac{s_T}{m}\right)^{mT}
}
\]

The present value of a future cash flow is:

\[
PV(CF_T)=CF_T\times DF(0,T)
\]

The corresponding spot rate can be recovered from the discount factor:

\[
s_T=
m\left[
DF(0,T)^{-\frac{1}{mT}}-1
\right]
\]

---

## 14. Zero-Coupon Yield Curve

A zero-coupon yield curve maps maturity to spot rate:

\[
T\longrightarrow s_T
\]

Every cash flow must be discounted using the spot rate corresponding to its maturity.

A yield curve may be:

- Upward sloping.
- Flat.
- Inverted.
- Humped.

The project represents every curve point with:

- Maturity.
- Spot rate.
- Discount factor.

---

## 15. Linear Interpolation

If a required maturity is located between two observed maturities, the implementation uses linear interpolation.

For:

\[
T_1<T<T_2
\]

the interpolated rate is:

\[
s(T)
=
s(T_1)
+
\frac{T-T_1}{T_2-T_1}
\left[
s(T_2)-s(T_1)
\right]
\]

Linear interpolation is transparent and easy to reproduce, although production fixed-income systems may use cubic splines, monotone convex methods, or parametric curve models.

The implementation does not extrapolate outside the observed maturity range.

---

## 16. Forward Rates

A forward rate is the future rate implied by today's spot-rate curve.

For two maturities \(T_1<T_2\):

\[
\left(1+\frac{s_{T_2}}{m}\right)^{mT_2}
=
\left(1+\frac{s_{T_1}}{m}\right)^{mT_1}
\left(1+\frac{f_{T_1,T_2}}{m}\right)^{m(T_2-T_1)}
\]

Solving for the forward rate:

\[
f_{T_1,T_2}
=
m\left[
\left(
\frac{
\left(1+\frac{s_{T_2}}{m}\right)^{mT_2}
}{
\left(1+\frac{s_{T_1}}{m}\right)^{mT_1}
}
\right)^{\frac{1}{m(T_2-T_1)}}
-1
\right]
\]

Forward rates satisfy a no-arbitrage relationship between investments over different horizons.

They are implied rates and should not automatically be interpreted as unbiased predictions.

---

## 17. Yield-Curve Bootstrapping

Bootstrapping extracts discount factors and spot rates sequentially from coupon-bearing bond prices.

For a bond maturing in period \(n\):

\[
P_n
=
C\sum_{i=1}^{n-1}DF_i
+
(F+C)DF_n
\]

All discount factors before \(DF_n\) are already known.

Therefore:

\[
DF_n
=
\frac{
P_n-C\sum_{i=1}^{n-1}DF_i
}{
F+C
}
\]

The corresponding spot rate is:

\[
s_n=
m\left[
DF_n^{-1/n}-1
\right]
\]

### Bootstrapping Algorithm

1. Sort instruments by maturity.
2. Begin with the shortest maturity.
3. Calculate its discount factor.
4. Use previously calculated discount factors to value earlier coupons.
5. Isolate the new maturity's discount factor.
6. Convert the discount factor into a spot rate.
7. Continue until the complete curve is constructed.

The implementation requires consecutive coupon periods and a consistent payment frequency.

---

## 18. Mean-Reverting Interest Rates

Short-term interest rates are modeled as mean-reverting processes.

Mean reversion reflects the idea that rates may move away from a long-term equilibrium but tend to return toward it.

The general drift component is:

\[
\kappa(\theta-r_t)
\]

where:

- \(\kappa>0\) is the speed of mean reversion.
- \(\theta\) is the long-term mean.
- \(r_t\) is the current short rate.

If:

\[
r_t<\theta
\]

the drift is positive.

If:

\[
r_t>\theta
\]

the drift is negative.

---

## 19. Vasicek Model

The Vasicek model is defined by:

\[
dr_t=
\kappa(\theta-r_t)dt+\sigma dW_t
\]

where:

- \(r_t\) is the short-term interest rate.
- \(\kappa\) is the mean-reversion speed.
- \(\theta\) is the long-term mean.
- \(\sigma\) is the volatility.
- \(W_t\) is a standard Brownian motion.

### Expected Rate

\[
E[r_t]
=
\theta+
(r_0-\theta)e^{-\kappa t}
\]

### Variance

\[
\operatorname{Var}(r_t)
=
\frac{\sigma^2}{2\kappa}
\left(
1-e^{-2\kappa t}
\right)
\]

### Euler Discretization

\[
r_{t+\Delta t}
=
r_t
+
\kappa(\theta-r_t)\Delta t
+
\sigma\sqrt{\Delta t}Z_t
\]

where:

\[
Z_t\sim N(0,1)
\]

Because the diffusion term is constant, Vasicek can generate negative rates.

---

## 20. Cox-Ingersoll-Ross Model

The CIR model is defined by:

\[
dr_t=
\kappa(\theta-r_t)dt
+
\sigma\sqrt{r_t}\,dW_t
\]

The square-root diffusion reduces volatility when rates approach zero.

### Expected Rate

\[
E[r_t]
=
\theta+
(r_0-\theta)e^{-\kappa t}
\]

### Variance

\[
\operatorname{Var}(r_t)
=
\frac{
r_0\sigma^2e^{-\kappa t}
}{
\kappa
}
\left(
1-e^{-\kappa t}
\right)
+
\frac{
\theta\sigma^2
}{
2\kappa
}
\left(
1-e^{-\kappa t}
\right)^2
\]

### Feller Condition

\[
2\kappa\theta\geq\sigma^2
\]

When this condition is satisfied, the theoretical continuous-time process remains strictly positive.

### Full-Truncation Euler Scheme

The implementation uses:

\[
r_t^+=\max(r_t,0)
\]

\[
\widetilde{r}_{t+\Delta t}
=
r_t
+
\kappa(\theta-r_t^+)\Delta t
+
\sigma\sqrt{r_t^+}
\sqrt{\Delta t}Z_t
\]

\[
r_{t+\Delta t}
=
\max(
\widetilde{r}_{t+\Delta t},
0
)
\]

This numerical procedure prevents negative simulated CIR rates.

---

## 21. Monte Carlo Simulation

For each short-rate model, the engine:

1. Defines a time grid.
2. Generates independent standard-normal shocks.
3. Initializes every path at the same short rate.
4. Applies the discretized stochastic equation.
5. Repeats the process across all time steps.
6. Stores the complete matrix of simulated paths.

The simulation matrix has dimensions:

\[
(\text{time steps}+1)
\times
(\text{number of simulations})
\]

A fixed random seed provides reproducibility.

Terminal-rate statistics include:

- Mean.
- Median.
- Standard deviation.
- Fifth percentile.
- Ninety-fifth percentile.
- Minimum.
- Maximum.

Percentile bands summarize model uncertainty over time.

---

## 22. Numerical Methods

### Brent Root Solver

Yield to Maturity is calculated through Brent's method.

The method combines characteristics of:

- Bisection.
- Secant iteration.
- Inverse quadratic interpolation.

It is robust when the root is bracketed and the pricing function is continuous.

### Euler Discretization

Vasicek and CIR paths are approximated using Euler time stepping.

Smaller time steps generally reduce discretization error but increase computational cost.

### Random Number Generation

The implementation uses NumPy's modern random-number generator:

```python
numpy.random.default_rng(seed)