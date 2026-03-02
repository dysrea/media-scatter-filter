import cv2
import numpy as np
import cv2.ximgproc

def get_dark_channel(image, patch_size=15):
    
    # Get minimum value across 3 color channels (axis=2)
    min_channels = np.min(image, axis=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    
    # Use erosion to find minimum value 
    dark_channel = cv2.erode(min_channels, kernel)
    
    return dark_channel

def get_atmospheric_light(image, dark_channel):
    
    h, w = image.shape[:2]
    image_size = h * w
    
    num_pixels = int(max(image_size * 0.001, 1))
    
    # Flatten arrays to 1D 
    dark_flat = dark_channel.reshape(image_size)
    image_flat = image.reshape(image_size, 3)
    
    # Sort dark channel pixels from darkest to brightest, 
    # then reverse it ([::-1]) to get brightest first. 
    indices = np.argsort(dark_flat)[::-1]
    
    # Grab coordinates of top 0.1% foggiest pixels
    top_indices = indices[:num_pixels]
    
    # Use coordinates to look at original color image
    top_pixels = image_flat[top_indices]
    
    # Atmospheric Light (A) is the maximum color value 
    A = np.max(top_pixels, axis=0)
    
    return A

def get_transmission(image, A, omega=0.95, patch_size=15):
    
    # Divide foggy image by A
    normalized_img = np.empty_like(image)
    for ind in range(3):
        normalized_img[:, :, ind] = image[:, :, ind] / A[ind]
    
    dark_normalized = get_dark_channel(normalized_img, patch_size)
    
    # (1 - omega * dark_channel)
    transmission = 1 - omega * dark_normalized
    
    return transmission

def recover_image(image, transmission, A, t0=0.1):
    
    res = np.empty_like(image)

    # Prevents division-by-zero artifacts
    t_clamped = cv2.max(transmission, t0)
    
    # Apply formula to all 3 channels
    for ind in range(3):
        res[:, :, ind] = (image[:, :, ind] - A[ind]) / t_clamped + A[ind]
        
    res = np.clip(res, 0, 1)
    
    return res

def guided_filter(I, p, radius=60, eps=0.0001):
    """
    I: The Guide Image (Grayscale original photo)
    p: The Filtering Input (Blocky transmission map)
    radius: Size of the window to look at (larger = more smoothing)
    eps: Regularization parameter (how sharp an edge needs to be to avoid blurring)
    """
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius))
    
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    
    # 'a' and 'b'
    # eps prevents division by zero and controls edge sharpness
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
    
    q = mean_a * I + mean_b
    
    return q

if __name__ == "__main__":
    print("Initializing pipeline...")
    
    img = cv2.imread('images/foggy.jpg').astype('float64') / 255.0
    
    dark_for_A = get_dark_channel(img, patch_size=15)
    A = get_atmospheric_light(img, dark_for_A)
    
    t_raw = get_transmission(img, A, patch_size=3, omega=0.90)
    
    # 8-Bit Guided Filter
    print("Refining depth map...")
    guide_8u = (img * 255).astype(np.uint8)
    gray_guide = cv2.cvtColor(guide_8u, cv2.COLOR_BGR2GRAY)
    t_raw_8u = (t_raw * 255).astype(np.uint8)
    
    # radius=60 to smooth tiny patches, eps=50 for sharp tree edges
    gf = cv2.ximgproc.createGuidedFilter(guide=gray_guide, radius=60, eps=50)
    t_refined_8u = gf.filter(t_raw_8u)
    
    t_refined = t_refined_8u.astype('float64') / 255.0
    
    clear_img = recover_image(img, t_refined, A)        
    print("Dehazing complete.")
    
    cv2.namedWindow('Final Clear Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Final Clear Image', 1000, 800)
    cv2.imshow('Final Clear Image', clear_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()