import { Milliseconds } from '../enums/Milliseconds.js';
import { browser } from './browser.js';

export const page = await browser.newPage();
page.setDefaultNavigationTimeout(Milliseconds.minute);
page.setDefaultTimeout(Milliseconds.minute);