import dayjs from 'dayjs';
import { parse } from 'node-html-parser';
import { page } from '../constants/page.js';
import { pageDelay } from '../constants/pageDelay.js';
import { Handed } from '../enums/Handed.js';
import type { PlayerTable } from '../interfaces/tables/PlayerTable.js';
import { getString } from './getString.js';
import { getPlayer } from './queries/getPlayer.js';
import { insertPlayer } from './queries/insertPlayer.js';
import { removeDiacritics } from './removeDiacritics.js';
import { wait } from './wait.js';

export const retrievePlayer = async (baseballReferenceId: string) => {
   const getBats = () => {
      const meta = dom.querySelector('#meta');
      const index = meta?.querySelector('.nothumb') ? 0 : 1;
      const metaDiv = meta?.querySelectorAll('div')[index];
      const handednessPs = metaDiv?.querySelectorAll('p');
      if (!handednessPs) {
         console.log('No handedness p tags while getting bats');
         return false;
      }
      const handednessP = handednessPs.find(handednessP => handednessP.innerText.includes('Bats:'));
      const handednessPieces = handednessP?.innerHTML.split('</strong>');
      if (!handednessPieces) {
         console.log('No handedness pieces while getting bats');
         return false;
      }
      const bats = handednessPieces[1].split('\n')[0].toLowerCase() as keyof typeof Handed;
      if (!Object.keys(Handed).includes(bats)) {
         console.log('handednessPieces[1]', handednessPieces[1]);
         console.log(`No Handed key for batting ${bats}`);
         return false;
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
         console.log('No handedness p tags while getting throws');
         return false;
      }
      const handednessP = handednessPs.find(handednessP => handednessP.innerText.includes('Bats:'));
      const handednessPieces = handednessP?.innerHTML.split('</strong>');
      if (!handednessPieces) {
         console.log('No handedness pieces while getting throws');
         return false;
      }
      const throws = handednessPieces[2].split('\n')[0].toLowerCase() as keyof typeof Handed;
      if (!Object.keys(Handed).includes(throws)) {
         console.log(`No Handed key for throwing ${throws}`);
         return false;
      }
      return Handed[throws];
   }

   const getTimeBorn = () => {
      const birthString = dom.querySelector('#necro-birth')?.getAttribute('data-birth');
      return dayjs(birthString).utc(true).unix();
   }

   const { rows: player } = await getPlayer(baseballReferenceId) as { rows: PlayerTable[] };
   if (player.length) {
      //console.log('player:', player[0]);
      return player[0];
   }
   await wait(pageDelay);
   const url = `https://www.baseball-reference.com/players/${baseballReferenceId}.shtml`;
   await page.goto(url, { waitUntil: 'domcontentloaded' });
   const html = await page.content();
   const dom = parse(html);
   const bats = getBats();
   if (bats === false)
      return false;
   const throws = getThrows();
   if (throws === false)
      return false;
   const timeBorn = getTimeBorn();
   const name = getName();
   console.log('inserting player', {
      baseball_reference_id: baseballReferenceId,
      bats,
      name,
      throws,
      time_born: timeBorn,
   })
   const { rows: newPlayer } = await insertPlayer({
      baseball_reference_id: baseballReferenceId,
      bats,
      name,
      throws,
      time_born: timeBorn,
   }) as { rows: PlayerTable[] };
   return newPlayer[0];
}