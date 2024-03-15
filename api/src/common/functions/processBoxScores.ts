import type { HTMLElement } from 'node-html-parser';
import { parse } from 'node-html-parser';
import type { Page } from 'puppeteer';
import { Milliseconds } from '../enums/Milliseconds.js';
import type { Player } from '../interfaces/tables/Player.js';
import type { WebBoxscore } from '../interfaces/tables/WebBoxscore.js';
import { getString } from './getString.js';
import { getOldestUnprocessedBoxscore } from './queries/getOldestUnprocessedBoxscore.js';
import { retrieveGame } from './retrieveGame.js';
import { retrievePlayer } from './retrievePlayer.js';
import { sleep } from './sleep.js';

export const processBoxScores = async (page: Page) => {
   const extractGameData = async (dom: HTMLElement, url: string) => {
      const baseballReferenceId = getString(url.split('/').slice(4).join('/').split('.').shift());
      return await retrieveGame(baseballReferenceId, dom);
   }

   const extractPlayerData = async (dom: HTMLElement) => {
      const playerTableHeaders = dom.querySelectorAll('th[data-stat="player"]');
      const baseballReferenceIds: string[] = [];
      playerTableHeaders.map(async playerTableHeader => {
         const a = playerTableHeader.querySelector('a');
         if (!a)
            return;
         const baseballReferenceId = getString(
            a?.getAttribute('href')?.replace('/players/', '').replace('.shtml', '')
         );
         if (!baseballReferenceIds.includes(baseballReferenceId))
            baseballReferenceIds.push(baseballReferenceId);
      })
      baseballReferenceIds.sort();
      return await retrievePlayers(baseballReferenceIds, []);
   }

   const extractPlayByPlayData = async () => {
      const table = dom.querySelector('#play_by_play');
      const tbody = table?.querySelector('tbody');
      const trs = tbody?.querySelectorAll('> *');
      trs?.forEach(tr => {
         const trId = tr.getAttribute('id');
         if (!trId?.startsWith('event_'))
            return;
         const tds = tr.querySelectorAll('td');

      })
   }

   const retrievePlayers = async (baseballReferenceIds: string[], players: Player[]): Promise<Player[]> => {
      if (!baseballReferenceIds.length)
         return players;
      const baseballReferenceId = baseballReferenceIds.shift();
      if (!baseballReferenceId)
         return players;
      const player = await retrievePlayer(baseballReferenceId, page);
      if (player !== false)
         players.push(player);
      await sleep(4 * Milliseconds.second);
      return await retrievePlayers(baseballReferenceIds, players);
   }

   const { rows: boxscores } = await getOldestUnprocessedBoxscore() as { rows: WebBoxscore[] };
   if (!boxscores.length)
      return;
   const { html, season, url } = boxscores[0];
   if (!html)
      return;
   const dom = parse(html);
   const gameData = await extractGameData(dom, url);
   if (gameData === false)
      return;
   const playerData = await extractPlayerData(dom);
}