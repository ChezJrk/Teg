/*
 Generic array container for both C and CUDA-C modes
 */

#ifndef GENERIC_ARRAY_H
#define GENERIC_ARRAY_H

template<typename T, int N>
struct generic_array {
    T values[N];

    TEG_DEVICE_HOST generic_array(){}
    TEG_DEVICE_HOST generic_array(T val) {
        for(int i = 0; i < N; i++)
            values[i] = val;
    }

    TEG_DEVICE_HOST T& operator[](int i){
        return values[i];
    }
};

template<typename T, int N>
TEG_DEVICE_HOST generic_array<T,N> operator*(generic_array<T,N> a, T b){
    generic_array<T,N> out;
    for(int i = 0; i < N; i++)
        out[i] = a[i] * b;
    return out;
}

template<typename T, int N>
TEG_DEVICE_HOST generic_array<T,N> operator*(T b, generic_array<T,N> a){
    generic_array<T,N> out;
    for(int i = 0; i < N; i++)
        out[i] = a[i] * b;
    return out;
}

template<typename T, int N>
TEG_DEVICE_HOST generic_array<T,N> operator*(generic_array<T,N> b, generic_array<T,N> a){
    generic_array<T,N> out;
    for(int i = 0; i < N; i++)
        out[i] = a[i] * b[i];
    return out;
}

template<typename T, int N>
TEG_DEVICE_HOST generic_array<T,N> operator+(T b, generic_array<T,N> a){
    generic_array<T,N> out;
    for(int i = 0; i < N; i++)
        out[i] = a[i] + b;
    return out;
}

template<typename T, int N>
TEG_DEVICE_HOST generic_array<T,N> operator+(generic_array<T,N> a, T b){
    generic_array<T,N> out;
    for(int i = 0; i < N; i++)
        out[i] = a[i] + b;
    return out;
}

template<typename T, int N>
TEG_DEVICE_HOST generic_array<T,N> operator+(generic_array<T,N> a, generic_array<T,N> b){
    generic_array<T,N> out;
    for(int i = 0; i < N; i++)
        out[i] = a[i] + b[i];
    return out;
}

#endif