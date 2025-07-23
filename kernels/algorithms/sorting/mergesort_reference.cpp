/*
 * Merge Sort - C++ Reference Implementation
 * 
 * Simple, single-threaded CPU implementation of merge sort
 * for educational purposes and correctness validation.
 * 
 * Algorithm:
 * - Divide array into halves recursively
 * - Sort each half
 * - Merge sorted halves
 * - Stable sort (preserves relative order)
 * - O(n log n) time, O(n) space
 */

#include <vector>
#include <algorithm>
#include <iostream>

/**
 * Merge two sorted subarrays
 */
void merge(std::vector<float>& arr, int left, int mid, int right) {
    std::vector<float> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    
    for (i = left, k = 0; i <= right; i++, k++) {
        arr[i] = temp[k];
    }
}

/**
 * Recursive merge sort
 */
void mergesort_recursive(std::vector<float>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergesort_recursive(arr, left, mid);
        mergesort_recursive(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

/**
 * C-style interface matching CUDA API
 */
int mergesort_reference(float* data, int n) {
    std::vector<float> arr(data, data + n);
    mergesort_recursive(arr, 0, n - 1);
    std::copy(arr.begin(), arr.end(), data);
    return 0;
} 