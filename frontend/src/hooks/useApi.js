/**
 * Custom hook for API calls with automatic error handling and loading states.
 * 
 * This hook provides a consistent way to make API calls across the application,
 * reducing code duplication and improving error handling.
 */

import { useState, useCallback } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const API = `${BACKEND_URL}/api`;

export const useApi = () => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const request = useCallback(async (endpoint, options = {}) => {
        const {
            method = 'GET',
            data = null,
            params = {},
            showErrorToast = true,
            onSuccess = null,
            onError = null
        } = options;

        setLoading(true);
        setError(null);

        try {
            const config = {
                method,
                url: `${API}${endpoint}`,
                params,
                ...(data && { data })
            };

            const response = await axios(config);

            if (onSuccess) {
                onSuccess(response.data);
            }

            return { data: response.data, error: null };
        } catch (err) {
            const errorMessage = err.response?.data?.detail || err.message || 'An error occurred';
            setError(errorMessage);

            if (showErrorToast) {
                toast.error(errorMessage);
            }

            if (onError) {
                onError(err);
            }

            return { data: null, error: errorMessage };
        } finally {
            setLoading(false);
        }
    }, []);

    return { request, loading, error };
};

/**
 * Custom hook for GET requests with caching support.
 */
export const useFetch = (endpoint, options = {}) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const { request } = useApi();

    const fetchData = useCallback(async () => {
        setLoading(true);
        const result = await request(endpoint, { ...options, showErrorToast: false });

        if (result.data) {
            setData(result.data);
        }
        if (result.error) {
            setError(result.error);
        }
        setLoading(false);
    }, [endpoint, request, options]);

    const refetch = useCallback(() => {
        return fetchData();
    }, [fetchData]);

    return { data, loading, error, refetch };
};

export default useApi;
