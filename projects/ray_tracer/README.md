# Ray Tracer

A complete implementation of "Ray Tracing in One Weekend" by Peter Shirley, featuring physically-based rendering with diffuse materials, metals, glass, and advanced camera controls. 

## Directory Structure

```
ray_tracer/
├── README.md               # This file
└── cpp/                    # Complete C++ implementation
    ├── vec3.h              # 3D vector mathematics with operator overloading
    ├── ray.h               # Ray class (origin + direction)
    ├── color.h             # Color utilities and PPM output functions
    ├── camera.h            # Camera system with positioning and depth of field
    ├── hittable.h          # Abstract hittable interface and hit records
    ├── shapes.h            # Sphere geometry implementation
    ├── material.h          # Material system (lambertian, metal, dielectric)
    ├── rtweekend.h         # Common utilities, constants, and random functions
    ├── ray_tracer.cpp      # Main program with configurable scenes
    ├── ray_tracer          # Compiled executable
    └── images/             # Generated ray-traced images (PPM format)
        ├── scene1_all_diffuse.ppm
        ├── scene2_perfect_metals.ppm
        ├── scene3_fuzzy_comparison.ppm
        ├── scene4_glass_materials.ppm
        ├── scene5_hollow_glass.ppm
        ├── scene6_positionable_camera.ppm
        └── scene7_defocus_blur.ppm
```

## Building and Running

```bash
cd cpp
g++ -o ray_tracer ray_tracer.cpp
./ray_tracer > images/output.ppm
```

**No external dependencies required** - uses only standard C++ libraries.

## Features Implemented

• **Vector Mathematics (`vec3.h`)** - Complete 3D vector class with operator overloading for arithmetic, dot/cross products, and utility functions. Supports all mathematical operations needed for ray tracing.

• **Ray Casting (`ray.h`)** - Ray class representing rays as P(t) = A + tB with origin and direction. Forms the foundation for all intersection and rendering calculations.

• **Camera System (`camera.h`)** - Flexible camera with positionable viewpoint, configurable field of view, and depth of field effects. Uses look-at geometry with arbitrary up vectors for full 3D positioning.

• **Object Intersection (`shapes.h`, `hittable.h`)** - Abstract hittable interface with sphere implementation using optimized quadratic formula. Supports ray-object intersection with surface normal calculation and front/back face detection.

• **Material System (`material.h`)** - Polymorphic material architecture supporting multiple surface types through virtual scatter functions. Materials control how light bounces off surfaces with physically-based parameters.

• **Diffuse Materials (Lambertian)** - Realistic matte surfaces using random hemisphere scattering with proper energy conservation. Implements true Lambertian reflection for natural-looking diffuse lighting.

• **Metal Materials** - Reflective surfaces with configurable fuzziness parameter (0.0 = perfect mirror, 1.0 = brushed metal). Uses reflection formula v - 2⟨v,n⟩n with random perturbation for surface roughness.

• **Dielectric Materials (Glass)** - Transparent materials with refraction and reflection using Snell's law and Schlick's approximation. Handles total internal reflection and provides realistic glass appearance with proper fresnel effects.

• **Antialiasing** - Monte Carlo sampling with configurable samples per pixel. Reduces aliasing through random ray sampling within pixel boundaries for smooth, high-quality images.

• **Recursive Ray Tracing** - Depth-limited recursive rendering supporting multiple light bounces between objects. Enables global illumination effects like color bleeding and realistic inter-reflection.

• **Defocus Blur (Depth of Field)** - Camera aperture simulation with focus distance control. Samples rays from defocus disk to create realistic depth of field effects with configurable blur strength.

• **Scene Management** - Flexible scene composition with multiple objects and materials. Supports complex scenes with various material combinations and spatial arrangements.

**Output:** PPM images (400×225, 16:9 aspect ratio) with up to 100 samples per pixel and 50 ray bounce depth for photorealistic rendering.
