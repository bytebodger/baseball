import dayjs from 'dayjs';
import { parse } from 'node-html-parser';
import { page } from '../constants/page.js';
import { pageDelay } from '../constants/pageDelay.js';
import { Handed } from '../enums/Handed.js';
import type { Result } from '../interfaces/Result.js';
import type { PlayerTable } from '../interfaces/tables/PlayerTable.js';
import { getString } from './getString.js';
import { getDBPlayer } from './queries/getDBPlayer.js';
import { insertDBPlayer } from './queries/insertDBPlayer.js';
import { removeDiacritics } from './removeDiacritics.js';
import { wait } from './wait.js';

export const scrapePlayer = async (baseballReferenceId: string, result: Result) => {
   const getBats = () => {
      const meta = dom.querySelector('#meta');
      const index = meta?.querySelector('.nothumb') ? 0 : 1;
      const metaDiv = meta?.querySelectorAll('div')[index];
      const handednessPs = metaDiv?.querySelectorAll('p');
      if (!handednessPs) {
         result.errors.push('No handedness p tags while getting bats');
         return;
      }
      const handednessP = handednessPs.find(handednessP => handednessP.innerText.includes('Bats:'));
      const handednessPieces = handednessP?.innerHTML.split('</strong>');
      if (!handednessPieces) {
         result.errors.push('No handedness pieces while getting bats');
         return;
      }
      const bats = handednessPieces[1].split('\n')[0].toLowerCase() as keyof typeof Handed;
      if (!Object.keys(Handed).includes(bats)) {
         result.errors.push(`No Handed key for batting ${bats}`);
         return;
      }
      return Handed[bats];
   }

   const getName = () => {
      const name = getString(dom.querySelector('h1 span')?.innerText);
      return removeDiacritics(name);
   }

   const getThrows = () => {
      const meta = dom.querySelector('#meta');
      const index = meta?.querySelector('.nothumb') ? 0 : 1;
      const metaDiv = meta?.querySelectorAll('div')[index];
      const handednessPs = metaDiv?.querySelectorAll('p');
      if (!handednessPs) {
         result.errors.push('No handedness p tags while getting throws');
         return;
      }
      const handednessP = handednessPs.find(handednessP => handednessP.innerText.includes('Bats:'));
      const handednessPieces = handednessP?.innerHTML.split('</strong>');
      if (!handednessPieces) {
         result.errors.push('No handedness pieces while getting throws');
         return;
      }
      const throws = handednessPieces[2].split('\n')[0].toLowerCase() as keyof typeof Handed;
      if (!Object.keys(Handed).includes(throws)) {
         result.errors.push(`No Handed key for throwing ${throws}`);
         return;
      }
      return Handed[throws];
   }

   const getTimeBorn = () => {
      const birthString = dom.querySelector('#necro-birth')?.getAttribute('data-birth');
      return dayjs(birthString).utc(true).unix();
   }

   const { rows: player } = await getDBPlayer(baseballReferenceId) as { rows: PlayerTable[] };
   if (player.length)
      return player[0];
   await wait(pageDelay);
   const url = `https://www.baseball-reference.com/players/${baseballReferenceId}.shtml`;
   await page.goto(url, { waitUntil: 'domcontentloaded' });
   const html = await page.content();
   const dom = parse(html);
   const bats = getBats();
   if (result.errors.length || !bats)
      return false;
   const throws = getThrows();
   if (result.errors.length || !throws)
      return false;
   const timeBorn = getTimeBorn();
   const name = getName();
   const fields = {
      baseball_reference_id: baseballReferenceId,
      bats,
      name,
      throws,
      time_born: timeBorn,
   }
   const { rows: newPlayer } = await insertDBPlayer(fields) as { rows: PlayerTable[] };
   result.messages.push('inserted player:');
   result.messages.push(fields);
   return newPlayer[0];
}