import dayjs from 'dayjs';
import { parse } from 'node-html-parser';
import type { Page } from 'puppeteer';
import { Handed } from '../enums/Handed.js';
import type { Player } from '../interfaces/tables/Player.js';
import { getString } from './getString.js';
import { getPlayer } from './queries/getPlayer.js';
import { insertPlayer } from './queries/insertPlayer.js';

export const retrievePlayer = async (baseballReferenceId: string, page: Page) => {
   const getBats = () => {
      const meta = dom.querySelector('#meta');
      const metaDiv = meta?.querySelectorAll('div')[1];
      const handednessP = metaDiv?.querySelectorAll('p')[1];
      const handednessPieces = handednessP?.innerHTML.split('</strong>');
      if (!handednessPieces) {
         console.log('No handedness pieces while getting bats');
         return false;
      }
      const bats = handednessPieces[1].split('\n')[0].toLowerCase() as keyof typeof Handed;
      if (!Object.keys(Handed).includes(bats)) {
         console.log(`No Handed key for ${bats}`);
         return false;
      }
      return Handed[bats];
   }

   const getName = () => {
      const h1 = dom.querySelector('h1');
      const h1Span = h1?.querySelector('span');
      return getString(h1Span?.innerText);
   }

   const getThrows = () => {
      const meta = dom.querySelector('#meta');
      const metaDiv = meta?.querySelectorAll('div')[1];
      const handednessP = metaDiv?.querySelectorAll('p')[1];
      const handednessPieces = handednessP?.innerHTML.split('</strong>');
      if (!handednessPieces) {
         console.log('No handedness pieces while getting throws');
         return false;
      }
      const throws = handednessPieces[2].split('\n')[0].toLowerCase() as keyof typeof Handed;
      if (!Object.keys(Handed).includes(throws)) {
         console.log(`No Handed key for ${throws}`);
         return false;
      }
      return Handed[throws];
   }

   const getTimeBorn = () => {
      const birthSpan = dom.querySelector('#necro-birth');
      const birthString = birthSpan?.getAttribute('data-birth');
      return dayjs(birthString).utc(true).valueOf();
   }

   const { rows: player } = await getPlayer(baseballReferenceId) as { rows: Player[] };
   if (player.length)
      return player[0];
   const url = `https://www.baseball-reference.com/players/${baseballReferenceId}.shtml`;
   await page.goto(url, { waitUntil: 'domcontentloaded' });
   const html = await page.content();
   const dom = parse(html);
   const name = getName();
   const bats = getBats();
   if (bats === false)
      return false;
   const throws = getThrows();
   if (throws === false)
      return false;
   const timeBorn = getTimeBorn();
   const { rows: newPlayer } = await insertPlayer({
      baseball_reference_id: baseballReferenceId,
      bats,
      name,
      throws,
      time_born: timeBorn,
   }) as { rows: Player[] };
   return newPlayer[0];
}