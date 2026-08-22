import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

export interface QueryResponse {
  answer: string;
}

export interface UploadResponse {
  chunks_ingested: number;
}

// Same-origin calls to the Django proxy that serves this app — Django forwards
// them to rag-api server-side.
@Injectable({ providedIn: 'root' })
export class RagApiService {
  constructor(private http: HttpClient) {}

  async query(question: string): Promise<QueryResponse> {
    return firstValueFrom(
      this.http.post<QueryResponse>('/api/query', { question })
    );
  }

  async uploadDocument(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return firstValueFrom(
      this.http.post<UploadResponse>('/api/ingest/upload', formData)
    );
  }
}
