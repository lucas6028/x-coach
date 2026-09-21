import { vi } from "vitest";

// A scripted XMLHttpRequest for the one request that uses it: api.analyzePose with an upload
// progress callback (fetch cannot report request-body progress). Installed with `vi.spyOn`, so a
// file's existing `afterEach(() => vi.restoreAllMocks())` removes it — no global left stubbed.

export interface FakeXhrScript {
  status?: number;
  body?: unknown;
  /** Upload progress events fired, in order, before the upload completes. */
  uploadEvents?: { loaded: number; total: number; lengthComputable?: boolean }[];
  /** Fail at the network level instead of answering. */
  networkError?: boolean;
}

export interface FakeXhrRequest {
  method: string;
  url: string;
  headers: Record<string, string>;
  body: FormData | null;
}

export function installFakeXhr(script: FakeXhrScript = {}): { requests: FakeXhrRequest[] } {
  const requests: FakeXhrRequest[] = [];
  vi.spyOn(globalThis, "XMLHttpRequest").mockImplementation(function FakeXhr() {
    const request: FakeXhrRequest = { method: "", url: "", headers: {}, body: null };
    const xhr = {
      status: 0,
      responseText: "",
      upload: {
        onprogress: null as ((e: unknown) => void) | null,
        onload: null as (() => void) | null,
      },
      onload: null as (() => void) | null,
      onerror: null as (() => void) | null,
      onabort: null as (() => void) | null,
      open(method: string, url: string) {
        request.method = method;
        request.url = url;
      },
      setRequestHeader(name: string, value: string) {
        request.headers[name] = value;
      },
      send(body: FormData) {
        request.body = body;
        requests.push(request);
        queueMicrotask(() => {
          if (script.networkError) {
            xhr.onerror?.();
            return;
          }
          for (const e of script.uploadEvents ?? []) {
            xhr.upload.onprogress?.({ lengthComputable: true, ...e });
          }
          xhr.upload.onload?.();
          xhr.status = script.status ?? 200;
          xhr.responseText = JSON.stringify(script.body ?? {});
          xhr.onload?.();
        });
      },
    };
    return xhr as unknown as XMLHttpRequest;
  } as unknown as () => XMLHttpRequest);
  return { requests };
}
