/**
 * SatQuery AI - API Service Layer
 * Centralized service for communicating with the ST-09 FastAPI backend.
 * Supports configurable backend URLs via localStorage and environment variables.
 */

export const DEFAULT_API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'https://accuracy-participate-updating-preceding.trycloudflare.com';

export const API_BASE_URL = DEFAULT_API_BASE_URL;

/**
 * Returns the currently active backend URL.
 * Checks localStorage first, then falls back to default.
 */
const OBSOLETE_ENDPOINTS = [
  'loca.lt',
  'honor-proceed-olympus-eleven',
  'mild-overseas-architecture-sally',
  'itchy-lands-kneel',
];

export function getApiBaseUrl() {
  if (typeof window !== 'undefined' && window.localStorage) {
    const custom = window.localStorage.getItem('satquery_backend_url');
    if (custom && custom.trim()) {
      const clean = custom.trim().replace(/\/+$/, '');
      const isObsolete = OBSOLETE_ENDPOINTS.some((pattern) => clean.includes(pattern));
      if (isObsolete) {
        window.localStorage.removeItem('satquery_backend_url');
        return DEFAULT_API_BASE_URL.replace(/\/+$/, '');
      }
      return clean;
    }
  }
  return DEFAULT_API_BASE_URL.replace(/\/+$/, '');
}

/**
 * Updates the backend URL in localStorage and broadcasts a change event.
 */
export function setApiBaseUrl(newUrl) {
  if (typeof window !== 'undefined' && window.localStorage) {
    const cleanUrl = (newUrl || '').trim().replace(/\/+$/, '');
    if (cleanUrl && cleanUrl !== DEFAULT_API_BASE_URL.replace(/\/+$/, '')) {
      window.localStorage.setItem('satquery_backend_url', cleanUrl);
    } else {
      window.localStorage.removeItem('satquery_backend_url');
    }
    window.dispatchEvent(new CustomEvent('satquery_backend_url_change', { detail: cleanUrl || DEFAULT_API_BASE_URL }));
  }
}

/**
 * Resets the backend URL to the application default.
 */
export function resetApiBaseUrl() {
  if (typeof window !== 'undefined' && window.localStorage) {
    window.localStorage.removeItem('satquery_backend_url');
    window.dispatchEvent(new CustomEvent('satquery_backend_url_change', { detail: DEFAULT_API_BASE_URL }));
  }
}

/**
 * Internal probe helper for /health endpoint.
 */
async function probeHealth(url, timeoutMs = 10000) {
  const baseUrl = (url || '').replace(/\/+$/, '');
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${baseUrl}/health`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
        'Bypass-Tunnel-Reminder': 'true',
      },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      return {
        isConnected: false,
        url: baseUrl,
        error: `Server returned HTTP ${response.status} (${response.statusText || 'Error'})`,
      };
    }

    const data = await response.json();
    const isHealthy =
      (data?.status === 'healthy' && data?.zoomearth === true && data?.grounding === true) ||
      data?.status === 'healthy' ||
      data?.service?.includes('SatQuery');

    return {
      isConnected: Boolean(isHealthy),
      url: baseUrl,
      data,
    };
  } catch (err) {
    clearTimeout(timeoutId);
    let errorMsg = err.message;
    if (err.name === 'AbortError') {
      errorMsg = 'Connection timed out (host took too long to respond)';
    } else if (err.message && err.message.includes('Failed to fetch')) {
      errorMsg = 'Could not resolve or reach backend host (tunnel may be offline or URL changed)';
    }

    return {
      isConnected: false,
      url: baseUrl,
      error: errorMsg,
    };
  }
}

/**
 * Probes the backend /health endpoint.
 * Accepts either the active URL or a custom URL for testing.
 * Automatically falls back to DEFAULT_API_BASE_URL if a saved custom URL is unreachable.
 */
export async function checkBackendHealth(customUrl = null) {
  const defaultUrl = DEFAULT_API_BASE_URL.replace(/\/+$/, '');
  const targetUrl = (customUrl || getApiBaseUrl()).replace(/\/+$/, '');

  // 1. Try the targeted URL
  const primaryResult = await probeHealth(targetUrl);
  if (primaryResult.isConnected) {
    return primaryResult;
  }

  // 2. If a saved custom URL failed, automatically attempt the default live URL
  if (!customUrl && targetUrl !== defaultUrl) {
    const fallbackResult = await probeHealth(defaultUrl);
    if (fallbackResult.isConnected) {
      // Clear out the unreachable custom URL so the client remains connected
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.removeItem('satquery_backend_url');
        window.dispatchEvent(new CustomEvent('satquery_backend_url_change', { detail: defaultUrl }));
      }
      return fallbackResult;
    }
  }

  return primaryResult;
}

/**
 * Sends satellite image and question to POST /analyze.
 * Automatically retries against default endpoint if a custom URL is unreachable.
 */
export async function analyzeRemoteSensingImage(imageFile, question) {
  if (!imageFile) {
    throw new Error('Please select an image file before analyzing.');
  }

  if (!question || !question.trim()) {
    throw new Error('Please enter a question to analyze.');
  }

  const defaultUrl = DEFAULT_API_BASE_URL.replace(/\/+$/, '');
  let baseUrl = getApiBaseUrl();
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('question', question.trim());

  let response;
  try {
    response = await fetch(`${baseUrl}/analyze`, {
      method: 'POST',
      headers: {
        'Bypass-Tunnel-Reminder': 'true',
      },
      body: formData,
    });
  } catch (networkErr) {
    // If custom URL failed, attempt fallback to default live endpoint
    if (baseUrl !== defaultUrl) {
      try {
        response = await fetch(`${defaultUrl}/analyze`, {
          method: 'POST',
          headers: {
            'Bypass-Tunnel-Reminder': 'true',
          },
          body: formData,
        });
        baseUrl = defaultUrl;
        if (typeof window !== 'undefined' && window.localStorage) {
          window.localStorage.removeItem('satquery_backend_url');
          window.dispatchEvent(new CustomEvent('satquery_backend_url_change', { detail: defaultUrl }));
        }
      } catch (_) {
        throw new Error(
          `Unable to reach backend at ${baseUrl} or default ${defaultUrl}. The tunnel URL may have expired or the server is offline.`
        );
      }
    } else {
      throw new Error(
        `Unable to reach backend at ${baseUrl}. The tunnel URL may have expired or the server is offline. Click "Configure Backend" in the top bar to update the endpoint URL.`
      );
    }
  }

  if (!response.ok) {
    let message = `Analysis request failed (HTTP ${response.status})`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        message =
          typeof errJson.detail === 'string'
            ? errJson.detail
            : JSON.stringify(errJson.detail);
      }
    } catch (_) {}
    throw new Error(message);
  }

  const raw = await response.json();

  // Normalize detections & objects
  const detections = Array.isArray(raw.detections)
    ? raw.detections
    : Array.isArray(raw.objects)
    ? raw.objects.map((obj) => ({
        label: obj.label,
        score: obj.confidence ?? obj.score ?? null,
        box: obj.box,
      }))
    : [];

  const isAbstained =
    raw.abstained === true ||
    raw.status === 'abstained' ||
    Boolean(raw.reason && !raw.answer);

  return {
    question: raw.question || question.trim(),
    answer: raw.answer || raw.reason || 'No text answer returned.',
    targets: Array.isArray(raw.targets) ? raw.targets : [],
    evidence_found: raw.evidence_found !== undefined ? raw.evidence_found : detections.length > 0,
    detections,
    abstained: isAbstained,
    evidence_image: raw.evidence_image || null,
    reason: raw.reason || null,
    status: raw.status || (isAbstained ? 'abstained' : 'answered'),
  };
}
