import type { HTMLElement } from 'node-html-parser';
import { parse } from 'node-html-parser';
import type { Page } from 'puppeteer';
import { Milliseconds } from '../enums/Milliseconds.js';
import type { Game } from '../interfaces/tables/Game.js';
import type { Player } from '../interfaces/tables/Player.js';
import type { WebBoxscore } from '../interfaces/tables/WebBoxscore.js';
import { getString } from './getString.js';
import { getOldestUnprocessedBoxscore } from './queries/getOldestUnprocessedBoxscore.js';
import { insertAtBat } from './queries/insertAtBat.js';
import { retrieveGame } from './retrieveGame.js';
import { retrievePlayer } from './retrievePlayer.js';
import { sleep } from './sleep.js';

export const processBoxScores = async (page: Page) => {
   const extractAtBats = async (game: Game, players: Player[]) => {
      const table = dom.querySelector('#play_by_play');
      const tbody = table?.querySelector('tbody');
      const trs = tbody?.querySelectorAll('> *');
      trs?.map(async tr => {
         const trId = tr.getAttribute('id');
         if (!trId?.startsWith('event_'))
            return;
         const tds = tr.querySelectorAll('td');
         const totalPitches = Number(tds[3].innerText.split(',').shift());
         const runs = Number(tds[4].innerText.match(/R/g)?.length);
         const outs = Number(tds[4].innerText.match(/O/g)?.length);
         const batterName = tds[6].innerText.replace(/&nbsp;/g, '');
         const batter = players.find(player => player.name === batterName);
         if (!batter) {
            console.log(`Could not find ${batterName} while getting at-bat`);
            return;
         }
         const pitcherName = tds[7].innerText.replace(/&nbsp;/g, '');
         const pitcher = players.find(player => player.name === pitcherName);
         if (!pitcher) {
            console.log(`Could not find ${pitcherName} while getting at-bat`);
            return;
         }
         const result = tds[10].innerText;
         let bases: number | null = null;
         if (
            result.includes('Double Play')
            || result.startsWith('Flyball')
            || result.startsWith('Foul Popfly')
            || result.startsWith('Groundout')
            || result.startsWith('Popfly')
            || result.startsWith('Lineout')
            || result.startsWith('Strikeout')
            || result.includes('Triple Play')
         )
            bases = 0;
         else if (
            result.startsWith('Hit By Pitch')
            || result.startsWith('Single to')
            || result.startsWith('Walk')
         )
            bases = 1;
         else if (
            result.startsWith('Double to')
            || result.startsWith('Ground-rule Double')
         )
            bases = 2;
         else if (result.startsWith('Triple to'))
            bases = 3;
         else if (result.startsWith('Home Run'))
            bases = 4;
         if (bases === null)
            return;
         await insertAtBat({
            bases,
            batter_player_id: batter.player_id,
            game_id: game.game_id,
            outs,
            pitcher_player_id: pitcher.player_id,
            runs,
            total_pitches: totalPitches,
         })
      })
   }

   const extractGame = async (dom: HTMLElement, url: string) => {
      const baseballReferenceId = getString(url.split('/').slice(4).join('/').split('.').shift());
      return await retrieveGame(baseballReferenceId, dom, page);
   }

   const extractPlayers = async (dom: HTMLElement) => {
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
      return await retrievePlayers(baseballReferenceIds, []);
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
   const { html, url } = boxscores[0];
   if (!html)
      return;
   const dom = parse(html);
   const game = await extractGame(dom, url);
   if (game === false)
      return;
   const players = await extractPlayers(dom);
   await extractAtBats(game, players);
}