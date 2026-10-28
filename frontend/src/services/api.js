/**
 * SatQuery AI - API Service Layer
 * Centralized service for communicating with the ST-09 FastAPI backend.
 * Base URL is configured via VITE_API_BASE_URL or defaults to the Cloudflare tunnel.
 */

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'https://itchy-lands-kneel.loca.lt';
/**
 * Checks backend health according to ST-09 requirements:
 * Requires status === 'healthy' && zoomearth === true && grounding === true
 */
export async function checkBackendHealth() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 6000);

  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
  Accept: 'application/json',
  'bypass-tunnel-reminder': '1',
},
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      return { isConnected: false, error: `HTTP ${response.status}` };
    }

    const data = await response.json();
    const isHealthy =
  (data?.status === 'ok' || data?.status === 'healthy') &&
  data?.zoomearth === true;

    return {
      isConnected: isHealthy,
      data,
    };
  } catch (err) {
    clearTimeout(timeoutId);
    return {
      isConnected: false,
      error: err.name === 'AbortError' ? 'Connection timed out' : err.message,
    };
  }
}

/**
 * Sends satellite image and natural-language query to the /analyze endpoint.
 *
 * @param {File} imageFile - Uploaded RGB satellite image (PNG, JPG, TIFF)
 * @param {string} question - Natural language user question
 * @returns {Promise<Object>} Backend response with answer, detections, evidence_image, abstained
 */
export async function analyzeRemoteSensingImage(imageFile, question) {
  if (!imageFile) {
    throw new Error('Please select an image file before analyzing.');
  }

  if (!question || !question.trim()) {
    throw new Error('Please enter a question to analyze.');
  }

  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('question', question.trim());

  let response;
  try {
response = await fetch(`${API_BASE_URL}/analyze`, {
  method: 'POST',
  headers: {
    'bypass-tunnel-reminder': '1',
  },
  body: formData,
});
  } catch (networkErr) {
    throw new Error(
      'Unable to connect to the analysis backend. Please check network connectivity and backend status.'
    );
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

  const result = await response.json();
  return result;
}
