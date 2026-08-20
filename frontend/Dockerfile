# syntax=docker/dockerfile:1
#
# x-coach frontend (React + Vite + Tailwind). Build context is frontend/.
#
# Two usable targets:
#   --target dev   Vite dev server with HMR (used by docker-compose.dev.yml)
#   (default)      static build served by nginx, which also proxies /api -> backend
#
# Vite inlines VITE_* at BUILD time, so the Supabase/LIFF values are build args, not
# runtime env. Rebuild the image to change them.

# --- deps: shared node_modules layer -------------------------------------------
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile


# --- dev: hot-reloading dev server ---------------------------------------------
FROM deps AS dev
ENV NODE_ENV=development
# Source is bind-mounted by compose; copying it keeps the target usable standalone.
COPY . .
EXPOSE 5173
# --host binds 0.0.0.0 so the port publish reaches it from outside the container.
CMD ["yarn", "dev", "--host", "0.0.0.0"]


# --- build: type-check + production bundle -------------------------------------
FROM deps AS build
COPY . .

ARG VITE_SUPABASE_URL=""
ARG VITE_SUPABASE_ANON_KEY=""
ARG VITE_LIFF_ID=""
ENV VITE_SUPABASE_URL=$VITE_SUPABASE_URL \
    VITE_SUPABASE_ANON_KEY=$VITE_SUPABASE_ANON_KEY \
    VITE_LIFF_ID=$VITE_LIFF_ID

RUN yarn build


# --- runtime: nginx ------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

# The config is templated so the backend it proxies to is a RUNTIME setting: compose keeps
# the service name, Azure Container Apps injects the backend's internal FQDN, and neither
# needs an image rebuild. nginx's entrypoint runs envsubst over /etc/nginx/templates/*.
#
# NGINX_ENVSUBST_FILTER is not optional. envsubst replaces every $NAME it sees, so without
# it $host, $remote_addr, $proxy_add_x_forwarded_for, $scheme and $uri would all be
# substituted away to empty strings -- a config that still starts and still proxies, but
# with blank forwarded headers and a broken SPA fallback. Pinning it to BACKEND_ORIGIN
# leaves nginx's own variables untouched. (Anchored at the start only: a trailing `$` in a
# Dockerfile ENV value is parsed as a build-arg reference, and no other variable in this
# image shares the prefix, so the closing anchor buys nothing worth the escaping.)
ENV BACKEND_ORIGIN=http://backend:8000 \
    NGINX_ENVSUBST_FILTER=^BACKEND_ORIGIN

COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
