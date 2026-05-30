import { Component, inject, OnInit, signal } from '@angular/core';
import { RouterLink } from '@angular/router';

import { ApiService } from '../../services/api.service';

type ApiStatus = 'loading' | 'online' | 'offline';

@Component({
  selector: 'app-home-page',
  imports: [RouterLink],
  templateUrl: './home.page.html',
  styleUrl: './home.page.scss'
})
export class HomePage implements OnInit {
  private readonly apiService = inject(ApiService);

  protected readonly apiStatus = signal<ApiStatus>('loading');

  ngOnInit(): void {
    this.apiService.getHealth().subscribe({
      next: () => this.apiStatus.set('online'),
      error: () => this.apiStatus.set('offline')
    });
  }
}
