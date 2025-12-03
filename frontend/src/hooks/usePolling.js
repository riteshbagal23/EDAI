/**
 * Custom hook for polling API endpoints at regular intervals.
 * 
 * This hook provides a reusable way to poll API endpoints, commonly used
 * for live monitoring, camera status updates, and detection updates.
 */

import { useEffect, useRef, useCallback } from 'react';
import { useApi } from './useApi';

export const usePolling = (endpoint, callback, interval = 3000, options = {}) => {
    const {
        enabled = true,
        onError = null,
        params = {}
    } = options;

    const { request } = useApi();
    const intervalRef = useRef(null);
    const isMountedRef = useRef(true);

    const poll = useCallback(async () => {
        if (!isMountedRef.current || !enabled) return;

        try {
            const result = await request(endpoint, {
                params,
                showErrorToast: false
            });

            if (result.data && isMountedRef.current) {
                callback(result.data);
            }

            if (result.error && onError) {
                onError(result.error);
            }
        } catch (error) {
            console.error('Polling error:', error);
            if (onError) {
                onError(error);
            }
        }
    }, [endpoint, callback, enabled, request, params, onError]);

    useEffect(() => {
        isMountedRef.current = true;

        if (!enabled) {
            return;
        }

        // Initial poll
        poll();

        // Set up interval
        intervalRef.current = setInterval(poll, interval);

        return () => {
            isMountedRef.current = false;
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [poll, interval, enabled]);

    const stop = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
    }, []);

    const start = useCallback(() => {
        stop();
        poll();
        intervalRef.current = setInterval(poll, interval);
    }, [poll, stop, interval]);

    return { stop, start };
};

export default usePolling;
