import { parse } from 'node-html-parser';
import type { Page } from 'puppeteer';
import { Milliseconds } from '../enums/Milliseconds.js';
import { wait } from './wait.js';

export const retrieveCoversOdds = async (date: string, visitor: string, host: string, page: Page) => {
   const getGameId = async () => {
      const url = `https://www.covers.com/sports/MLB/matchups?selectedDate=${date}`;
      await page.goto(url, { waitUntil: 'domcontentloaded' });
      const html = await page.content();
      const dom = parse(html);
      const gameBoxes = dom.querySelectorAll('div[data-game-date]:not([data-game-date=""])');
      const gameBox = gameBoxes.find(gameBox => {
         return gameBox.getAttribute('data-away-team-shortname-search') === visitor
            && gameBox.getAttribute('data-home-team-shortname-search') === host;
      })
      const dataLink = gameBox?.getAttribute('data-link');
      return Number(dataLink?.split('/').pop());
   }

   const getHostMoneyline = () => {
      const sponsoredOddsTable = dom.querySelector('#sponsoredOdds-table');
      const tbody = sponsoredOddsTable?.querySelector('tbody');
      const trs = tbody?.querySelectorAll('tr');
      if (!trs) {
         console.log('No tr tags while getting host moneyline');
         return false;
      }
      const tds = trs[1].querySelectorAll('td');
      const a = tds[1].querySelector('a');
      if (!a) {
         console.log('No a tag while getting host moneyline');
         return false;
      }
      return Number(a.getAttribute('data-market-american-odds'));
   }

   const getOverMoneyline = () => {
      const sponsoredOddsTable = dom.querySelector('#sponsoredOdds-table');
      const tbody = sponsoredOddsTable?.querySelector('tbody');
      const trs = tbody?.querySelectorAll('tr');
      if (!trs) {
         console.log('No tr tags while getting over moneyline');
         return false;
      }
      const tds = trs[0].querySelectorAll('td');
      const a = tds[3].querySelector('a');
      if (!a) {
         console.log('No a tag while getting over moneyline');
         return false;
      }
      return Number(a.getAttribute('data-market-american-odds'));
   }

   const getOverUnder = () => {
      const sponsoredOddsTable = dom.querySelector('#sponsoredOdds-table');
      const tbody = sponsoredOddsTable?.querySelector('tbody');
      const trs = tbody?.querySelectorAll('tr');
      if (!trs) {
         console.log('No tr tags while getting over-under');
         return false;
      }
      const tds = trs[0].querySelectorAll('td');
      const a = tds[3].querySelector('a');
      if (!a) {
         console.log('No a tag while getting over-under');
         return false;
      }
      return Number(a.getAttribute('data-handicap')?.replace('o', ''));
   }

   const getUnderMoneyline = () => {
      const sponsoredOddsTable = dom.querySelector('#sponsoredOdds-table');
      const tbody = sponsoredOddsTable?.querySelector('tbody');
      const trs = tbody?.querySelectorAll('tr');
      if (!trs) {
         console.log('No tr tags while getting under moneyline');
         return false;
      }
      const tds = trs[1].querySelectorAll('td');
      const a = tds[2].querySelector('a');
      if (!a) {
         console.log('No a tag while getting under moneyline');
         return false;
      }
      return Number(a.getAttribute('data-market-american-odds'));
   }

   const getVisitorMoneyline = () => {
      const sponsoredOddsTable = dom.querySelector('#sponsoredOdds-table');
      const tbody = sponsoredOddsTable?.querySelector('tbody');
      const trs = tbody?.querySelectorAll('tr');
      if (!trs) {
         console.log('No tr tags while getting visitor moneyline');
         return false;
      }
      const tds = trs[0].querySelectorAll('td');
      const a = tds[2].querySelector('a');
      if (!a) {
         console.log('No a tag while getting visitor moneyline');
         return false;
      }
      return Number(a.getAttribute('data-market-american-odds'));
   }

   const retrieveMatchupDom = async (gameId: number) => {
      const url = `https://www.covers.com/sport/baseball/mlb/matchup/${gameId}`;
      await page.goto(url, { waitUntil: 'domcontentloaded' });
      const html = await page.content();
      return parse(html);
   }

   const gameId = await getGameId();
   await wait(4 * Milliseconds.second);
   const dom = await retrieveMatchupDom(gameId);
   const visitorMoneyline = getVisitorMoneyline();
   if (visitorMoneyline === false)
      return false;
   const overUnder = getOverUnder();
   if (overUnder === false)
      return false;
   const overMoneyline = getOverMoneyline();
   if (overMoneyline === false)
      return false;
   const hostMoneyline = getHostMoneyline();
   if (hostMoneyline === false)
      return false;
   const underMoneyline = getUnderMoneyline();
   if (underMoneyline === false)
      return false;
   return {
      hostMoneyline,
      overMoneyline,
      overUnder,
      underMoneyline,
      visitorMoneyline,
   }
}