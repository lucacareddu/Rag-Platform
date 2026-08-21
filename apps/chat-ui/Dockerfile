# Stage 1: build the Angular app
FROM node:20-slim AS angular-build
WORKDIR /angular
COPY angular-app/package.json angular-app/package-lock.json ./
RUN npm ci
COPY angular-app .
RUN npm run build

# Stage 2: Django, serving the built Angular files + proxying to rag-api
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config ./config
COPY ragproxy ./ragproxy
COPY manage.py .
COPY --from=angular-build /angular/dist/chat-ui/browser ./static-app
RUN python manage.py collectstatic --noinput
EXPOSE 8080
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8080", "--workers", "2"]
