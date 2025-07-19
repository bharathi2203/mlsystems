/*
 * vec3.h - 3D Vector Mathematics
 * 
 * Implements a complete 3D vector class with operator overloading for arithmetic operations,
 * dot/cross products, and utility functions needed for ray tracing calculations.
 */

#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

/**
 * vec3 - 3D Vector Class
 * 
 * Represents a 3D vector with x, y, z components. Supports all standard vector operations
 * including arithmetic, dot product, cross product, normalization, and length calculations.
 * Used throughout the ray tracer for positions, directions, colors, and normals.
 */
class vec3 {
  public:
    double e[3];  // Array to store x, y, z components

    // Default constructor - initializes to zero vector
    vec3() { e[0] = e[1] = e[2] = 0; }
    
    // Parameterized constructor
    vec3(double e0, double e1, double e2) { e[0] = e0; e[1] = e1; e[2] = e2; }

    // Const accessor methods - don't modify the object
    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }

    // Unary minus operator - returns negated vector
    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    
    // Array subscript operators - const and non-const versions
    double operator[](int i) const { return e[i]; }
    double& operator[](int i) { return e[i]; }

    // Compound assignment operators - modify this object and return reference
    vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;  // Return reference to this object for chaining
    }

    vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vec3& operator/=(double t) {
        return *this *= 1/t;  // Delegate to *= operator
    }

    // Vector magnitude (length)
    double length() const {
        return std::sqrt(length_squared());
    }

    // Squared length (avoids sqrt for efficiency when comparing lengths)
    double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    /**
     * near_zero - Check if vector is close to zero in all dimensions
     * Used to detect degenerate scatter directions in material calculations
     */
    bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        const double s = 1e-8;
        return (std::abs(e[0]) < s) && (std::abs(e[1]) < s) && (std::abs(e[2]) < s);
    }
};

// Type alias for semantic clarity (C++98 compatible)
typedef vec3 point3;

// ============================================================================
// Vector Utility Functions (Free Functions)
// ============================================================================

// Stream output operator
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

// Binary arithmetic operators
inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

// Element-wise multiplication
inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

// Scalar multiplication (both orders)
inline vec3 operator*(double t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator*(const vec3& v, double t) {
    return t * v;  // Delegate to the other version
}

// Scalar division
inline vec3 operator/(const vec3& v, double t) {
    return (1/t) * v;
}

/**
 * dot - Dot Product
 * Formula: u·v = u.x*v.x + u.y*v.y + u.z*v.z
 * Returns scalar result, used for projection and angle calculations
 */
inline double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

/**
 * cross - Cross Product
 * Formula: u×v = (u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x)
 * Returns vector perpendicular to both input vectors
 */
inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

// Unit vector (normalized)
inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

/**
 * reflect - Vector Reflection
 * Formula: r = v - 2(v·n)n
 * Reflects vector v off surface with normal n (used for mirror materials)
 */
inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

/**
 * refract - Vector Refraction (Snell's Law)
 * Implements Snell's law: η₁sin(θ₁) = η₂sin(θ₂)
 * Used for transparent materials like glass
 */
inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    double cos_theta = std::min(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -std::sqrt(std::abs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif 