# Physics-Based Media Scattering Reversal for Remote Sensing

Most modern Computer Vision (CV) pipelines treat the camera as a black box, relying entirely on heavy Deep Learning models to "hallucinate" or fix degraded images. This project takes a fundamentally different approach: solving vision degradation at the **physics and optical layers**.

This pipeline mathematically models and reverses the scattering of photons in participating media (atmosphere, smoke, or turbid water). By calculating the exact physical degradation of the optical signal, it strips away interference before the image ever reaches a downstream perception algorithm. This makes it highly applicable for **Earth Observation satellites, autonomous drones, and remote sensing in extreme environments.**



## 1. The Optical Physics Model
The foundation of this pipeline is Koschmieder’s Law (1924), which defines the Atmospheric Scattering Model. When capturing an image over long distances, the observed light is a combination of the true scene radiance and the ambient light scattered into the lens by the medium.

$$I(x) = J(x)t(x) + A(1 - t(x))$$

Where:
* $I(x)$ : The observed degraded image (what the sensor captures).
* $J(x)$ : The true scene radiance (the clear optical signal we want to recover).
* $A$ : The Global Atmospheric Light (the color and intensity of the scattering medium).
* $t(x)$ : The Transmission Map. This describes the portion of light that survives the journey to the camera without scattering. 

In physics, transmission is exponentially related to scene depth $d(x)$ and the medium's scattering coefficient $\beta$:
$$t(x) = e^{-\beta d(x)}$$

## 2. The Algorithm & Mathematics
To solve for $J(x)$, we must estimate $A$ and $t(x)$ from a single image. This pipeline implements the **Dark Channel Prior (DCP)** (He et al., 2009), based on the statistical observation that in non-sky patches of an outdoor image, at least one color channel has some pixels with intensities very close to zero.

### Step 1: Extracting the Dark Channel
We compute the dark channel $J^{dark}(x)$ by taking the minimum filter across all three RGB color channels $c$, and then taking the local minimum within a spatial patch $\Omega(x)$:

$$J^{dark}(x) = \min_{c \in \{r,g,b\}} \left( \min_{y \in \Omega(x)} J^{c}(y) \right) \approx 0$$

### Step 2: Estimating Global Atmospheric Light ($A$)
To find the exact color and intensity of the optical interference ($A$), we cannot simply pick the brightest pixel in the image (which might be a white object or a specular reflection). Instead, we:
1. Isolate the top $0.1\%$ brightest pixels within the Dark Channel (representing the most optically dense scattering/furthest depth).
2. Find the corresponding pixels in the original image $I(x)$.
3. Take the maximum intensity among these pixels to define $A = [A_r, A_g, A_b]$.

### Step 3: Calculating the Transmission Map ($t(x)$)
By normalizing the scattering equation by $A$ and applying the dark channel operation on both sides, the true radiance term drops out (since $J^{dark} \approx 0$). We can then solve directly for the transmission map:

$$t(x) = 1 - \omega \min_{c} \left( \min_{y \in \Omega(x)} \frac{I^c(y)}{A^c} \right)$$

*Note: A constant $\omega = 0.85$ to $0.95$ is introduced to retain a small amount of distant scattering, maintaining natural depth perception and preventing color over-saturation ("Red Shift"). We use a micro-patch size (e.g., 3x3) here to prevent the "Fat Boundary Problem."*

### Step 4: Edge-Preserving Smoothing (The Guided Filter)
The raw transmission map $t(x)$ is calculated in discrete local patches, resulting in blocky artifacts. To refine this into a pixel-perfect depth map, we use a **Guided Filter**. 

The filter assumes the refined transmission $q$ is a linear transform of a high-resolution Guide Image $I$ (the original grayscale image) in a local window $\omega_k$:
$$q_i = a_k I_i + b_k, \quad \forall i \in \omega_k$$
By calculating the local variance and covariance matrices of the guide image, the linear coefficients $a_k$ and $b_k$ are determined. This smooths the blocky transmission map perfectly in flat areas (like sky/water) while slicing exactly along sharp physical edges (like structural silhouettes).

### Step 5: Scene Radiance Recovery
With $A$ and $t(x)$ mathematically estimated, we invert Koschmieder’s equation to recover the true optical signal $J(x)$:

$$J(x) = \frac{I(x) - A}{\max(t(x), t_0)} + A$$

*Note: The term $t_0$ (typically $0.1$) acts as a lower-bound threshold to prevent division-by-zero explosions in areas where the optical medium is infinitesimally dense.*

## Tech Stack & Execution
* **Language:** Python
* **Libraries:** NumPy (Vectorized matrix calculus), OpenCV (`cv2`, `ximgproc` for 8-bit Guided Filter optimization).