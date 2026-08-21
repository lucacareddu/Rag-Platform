import { Component, ElementRef, ViewChild, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RagApiService } from './services/rag-api.service';

type Role = 'user' | 'assistant' | 'error';

interface Message {
  role: Role;
  text: string;
}

type UploadState = 'idle' | 'uploading' | 'done' | 'error';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements AfterViewChecked {
  @ViewChild('scrollAnchor') private scrollAnchor?: ElementRef<HTMLDivElement>;
  @ViewChild('fileInput') private fileInput?: ElementRef<HTMLInputElement>;

  messages: Message[] = [];
  draft = '';
  sending = false;

  uploadState: UploadState = 'idle';
  uploadFileName = '';
  uploadResultText = '';

  private shouldScroll = false;

  constructor(private ragApi: RagApiService) {}

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      this.scrollAnchor?.nativeElement.scrollIntoView({ behavior: 'smooth' });
      this.shouldScroll = false;
    }
  }

  async send(): Promise<void> {
    const question = this.draft.trim();
    if (!question || this.sending) return;

    this.messages.push({ role: 'user', text: question });
    this.draft = '';
    this.sending = true;
    this.shouldScroll = true;

    try {
      const res = await this.ragApi.query(question);
      this.messages.push({ role: 'assistant', text: res.answer });
    } catch (e) {
      this.messages.push({
        role: 'error',
        text: 'The request failed. Check that rag-api is reachable and try again.',
      });
    } finally {
      this.sending = false;
      this.shouldScroll = true;
    }
  }

  onEnter(event: Event): void {
    const keyEvent = event as KeyboardEvent;
    if (!keyEvent.shiftKey) {
      keyEvent.preventDefault();
      this.send();
    }
  }

  triggerUpload(): void {
    this.fileInput?.nativeElement.click();
  }

  async onFileSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    this.uploadFileName = file.name;
    this.uploadState = 'uploading';

    try {
      const res = await this.ragApi.uploadDocument(file);
      this.uploadState = 'done';
      this.uploadResultText = `Added ${res.chunks_ingested} chunk${res.chunks_ingested === 1 ? '' : 's'} from ${file.name}.`;
    } catch (e) {
      this.uploadState = 'error';
      this.uploadResultText = `Couldn't ingest ${file.name}. Check the file type and try again.`;
    } finally {
      input.value = '';
    }
  }

  dismissUploadStatus(): void {
    this.uploadState = 'idle';
    this.uploadResultText = '';
    this.uploadFileName = '';
  }
}
