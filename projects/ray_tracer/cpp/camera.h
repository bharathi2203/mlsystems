/*
 * camera.h - Camera System with Positioning and Depth of Field
 * 
 * Implements a flexible camera system supporting:
 * - Arbitrary positioning with look-at geometry
 * - Configurable field of view
 * - Antialiasing through multi-sampling
 * - Depth of field effects with defocus blur
 * - Recursive ray tracing with material interactions
 */

#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"
#include "hittable.h"
#include "material.h"

/**
 * camera - Perspective Camera with Advanced Features
 * 
 * Provides a complete camera system for ray tracing with support for:
 * - 3D positioning using look-at geometry
 * - Configurable vertical field of view
 * - Multi-sample antialiasing
 * - Depth of field simulation
 * - Recursive ray tracing for global illumination
 */
class camera {
  public:
    double aspect_ratio;      // Ratio of image width over height
    int    image_width;       // Rendered image width in pixel count
    int    samples_per_pixel; // Count of random samples for each pixel
    int    max_depth;         // Maximum number of ray bounces into scene

    double vfov;              // Vertical view angle (field of view)
    point3 lookfrom;          // Point camera is looking from
    point3 lookat;            // Point camera is looking at
    vec3   vup;               // Camera-relative "up" direction

    double defocus_angle;     // Variation angle of rays through each pixel
    double focus_dist;        // Distance from camera lookfrom point to plane of perfect focus

    /**
     * Constructor - Initialize camera with default settings
     * Sets up reasonable defaults for all camera parameters
     */
    camera() {
        aspect_ratio = 16.0 / 9.0;
        image_width = 400;
        samples_per_pixel = 10;
        max_depth = 10;
        vfov = 90;
        lookfrom = point3(0,0,-1);
        lookat = point3(0,0,0);
        vup = vec3(0,1,0);
        defocus_angle = 0;
        focus_dist = 10;
    }

    /**
     * render - Main rendering function
     * 
     * Renders the scene to PPM format by casting rays through each pixel,
     * applying antialiasing, and performing recursive ray tracing.
     * 
     * @param world The scene to render (collection of hittable objects)
     */
    void render(const hittable& world) {
        initialize();

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = 0; j < image_height; j++) {
            std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; i++) {
                color pixel_color = color(0, 0, 0);
                
                // Take multiple samples per pixel for antialiasing
                for (int sample = 0; sample < samples_per_pixel; sample++) {
                    ray r = get_ray(i, j);
                    pixel_color += ray_color(r, max_depth, world);
                }
                
                write_color(std::cout, pixel_color * (1.0 / samples_per_pixel));
            }
        }

        std::clog << "\rDone.                 \n";
    }

  private:
    int    image_height;   // Rendered image height
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below
    vec3   u, v, w;        // Camera frame basis vectors
    vec3   defocus_disk_u; // Defocus disk horizontal radius
    vec3   defocus_disk_v; // Defocus disk vertical radius

    /**
     * initialize - Set up camera coordinate system and viewport
     * 
     * Calculates camera basis vectors (u,v,w) using look-at geometry,
     * sets up viewport dimensions based on field of view and focus distance,
     * and initializes defocus disk for depth of field effects.
     */
    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = lookfrom;

        // Determine viewport dimensions.
        double theta = degrees_to_radians(vfov);
        double h = tan(theta/2);
        double viewport_height = 2 * h * focus_dist;
        double viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        vec3 viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        double defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    /**
     * get_ray - Generate ray for given pixel with antialiasing and depth of field
     * 
     * Creates a ray from camera through specified pixel with random sampling
     * for antialiasing and depth of field effects.
     * 
     * @param i Pixel column index
     * @param j Pixel row index
     * @return Ray from camera through pixel with random perturbations
     */
    ray get_ray(int i, int j) const {
        // Get a randomly-sampled camera ray for the pixel at location i,j, originating from
        // the camera defocus disk.
        point3 pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        point3 pixel_sample = pixel_center + pixel_sample_square();

        point3 ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        vec3 ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    /**
     * pixel_sample_square - Random sampling within pixel for antialiasing
     * @return Random offset within pixel bounds
     */
    vec3 pixel_sample_square() const {
        // Returns a random point in the square surrounding a pixel at the origin.
        double px = -0.5 + random_double();
        double py = -0.5 + random_double();
        return (px * pixel_delta_u) + (py * pixel_delta_v);
    }

    /**
     * defocus_disk_sample - Random sampling from camera aperture for depth of field
     * @return Random point on defocus disk for depth of field effect
     */
    point3 defocus_disk_sample() const {
        // Returns a random point in the camera defocus disk.
        vec3 p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    /**
     * ray_color - Recursive ray tracing with material interactions
     * 
     * Traces a ray through the scene, handling material scattering and
     * recursive ray bouncing for global illumination effects.
     * 
     * @param r The ray to trace
     * @param depth Remaining bounce depth (prevents infinite recursion)
     * @param world Scene objects to test for intersection
     * @return Color contribution from this ray
     */
    color ray_color(const ray& r, int depth, const hittable& world) const {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0,0,0);

        hit_record rec;

        if (world.hit(r, interval(0.001, infinity), rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered))
                return attenuation * ray_color(scattered, depth-1, world);
            return color(0,0,0);
        }

        // Background gradient
        vec3 unit_direction = unit_vector(r.direction());
        double a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }
};

#endif
