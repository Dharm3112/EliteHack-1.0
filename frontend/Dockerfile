# ---- Build Stage ----
FROM node:18-alpine as build
WORKDIR /app

# Copy package configurations
COPY package*.json ./
RUN npm install

# Copy application code
COPY . .

# Build Vite application
RUN npm run build

# ---- Production Server Stage ----
FROM nginx:alpine

# Copy built artifacts from the build stage to Nginx web root
COPY --from=build /app/dist /usr/share/nginx/html

# Expose HTTP port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
