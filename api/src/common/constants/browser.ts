import puppeteer from 'puppeteer';

export const browser = await puppeteer.launch({
   ignoreHTTPSErrors: true,
   timeout: 0,
   protocol: 'cdp',
   protocolTimeout: 0,
})