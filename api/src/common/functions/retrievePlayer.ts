import dayjs from 'dayjs';
import { parse } from 'node-html-parser';
import type { Page } from 'puppeteer';
import { Handed } from '../enums/Handed.js';
import type { Player } from '../interfaces/tables/Player.js';
import { getPlayer } from './queries/getPlayer.js';
import { insertPlayer } from './queries/insertPlayer.js';

export const retrievePlayer = async (baseballReferenceId: string, page: Page) => {
   const { rows: player } = await getPlayer(baseballReferenceId) as { rows: Player[] };
   if (player.length)
      return player[0];
   const url = `https://www.baseball-reference.com/players/${baseballReferenceId}.shtml`;
   await page.goto(url, { waitUntil: 'domcontentloaded' });
   const html = await page.content();
   const dom = parse(html);
   const meta = dom.querySelector('#meta');
   const metaDiv = meta?.querySelectorAll('div')[1];
   const handednessP = metaDiv?.querySelectorAll('p')[1];
   const handednessPieces = handednessP?.innerHTML.split('</strong>');
   if (!handednessPieces)
      return false;
   const bats = handednessPieces[1].split('\n')[0].toLowerCase() as keyof typeof Handed;
   if (!Object.keys(Handed).includes(bats)) {
      console.log(`No Handed key for ${bats}`);
      return false;
   }
   const throws = handednessPieces[2].split('\n')[0].toLowerCase() as keyof typeof Handed;
   if (!Object.keys(Handed).includes(throws)) {
      console.log(`No Handed key for ${throws}`);
      return false;
   }
   const birthSpan = dom.querySelector('#necro-birth');
   const birthString = birthSpan?.getAttribute('data-birth');
   const timeBorn = dayjs(birthString).utc(true).valueOf();
   const { rows: newPlayer } = await insertPlayer({
      baseball_reference_id: baseballReferenceId,
      bats: Handed[bats],
      throws: Handed[throws],
      time_born: timeBorn,
   }) as { rows: Player[] };
   return {
      baseball_reference_id: baseballReferenceId,
      bats: Handed[bats],
      player_id: newPlayer[0].player_id,
      throws: Handed[throws],
      time_born: timeBorn,
   };
}