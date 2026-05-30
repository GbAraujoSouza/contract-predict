import { Routes } from '@angular/router';
import { HomePage } from './pages/home/home.page';
import { InfoPage } from './pages/info/info.page';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'home' },
  { path: 'home', component: HomePage },
  { path: 'info', component: InfoPage },
  { path: '**', redirectTo: 'home' }
];
