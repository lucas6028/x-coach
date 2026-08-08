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
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
