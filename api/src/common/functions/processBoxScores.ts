import dayjs from 'dayjs';
import type { HTMLElement } from 'node-html-parser';
import { parse } from 'node-html-parser';
import type { Page } from 'puppeteer';
import { Milliseconds } from '../enums/Milliseconds.js';
import { PlayingSurface } from '../enums/PlayingSurface.js';
import { Umpire } from '../enums/Umpire.js';
import { Venue } from '../enums/Venue.js';
import type { Player } from '../interfaces/tables/Player.js';
import type { WebBoxscore } from '../interfaces/tables/WebBoxscore.js';
import { getString } from './getString.js';
import { getOldestUnprocessedBoxscore } from './queries/getOldestUnprocessedBoxscore.js';
import { retrievePlayer } from './retrievePlayer.js';
import { sleep } from './sleep.js';

export const processBoxScores = async (page: Page) => {
   interface GameData {
      baseballReferenceId: string,
      dayOfYear: number,
      gameOfSeason: number,
      hourOfDay: number,
      playingSurface: PlayingSurface,
      temperature: number,
      umpire: Umpire,
      venue: Venue,
   }
   const extractGameData = (dom: HTMLElement, url: string) => {
      const baseballReferenceId = getString(url.split('/').slice(4).join('/').split('.').shift());
      const metaDivs = dom.querySelectorAll('.scorebox_meta > *');
      const gameDayString = metaDivs[0].innerText.split(',').slice(1).join(',').trim();
      const gameDay = dayjs(gameDayString).utc(true);
      const dayOfYear = gameDay.dayOfYear();
      const [time, amPm] = metaDivs[1].innerText.split(':').slice(1).join(':').trim().split(' ').slice(0, 2);
      let hourOfDay = Number(time.split(':').shift());
      if (amPm === 'a.m.' && hourOfDay === 12)
         hourOfDay = 24;
      else if (amPm === 'p.m.' && hourOfDay < 12)
         hourOfDay += 12;
      const venueDiv = metaDivs.find(metaDiv => metaDiv.innerHTML.includes('Venue'));
      const venue = getString(
         venueDiv?.innerHTML.split(':').pop()?.trim().replace('"', '')
      ) as keyof typeof Venue;
      if (!Object.keys(Venue).includes(venue)) {
         console.log(`No Venue key for ${venue}`);
         return false;
      }
      const surfaceDiv = metaDivs.find(metaDiv => metaDiv.innerHTML.includes(', on '));
      const playingSurface = getString(surfaceDiv?.innerHTML.split(', on ').pop()) as keyof typeof PlayingSurface;
      if (!Object.keys(PlayingSurface).includes(playingSurface)) {
         console.log(`No PlayingSurface key for ${playingSurface}`);
         return false;
      }
      const otherInfo = dom.querySelector('span[data-label="Other Info"]')?.parentNode.parentNode;
      const sectionContent = otherInfo?.querySelector('.section_content');
      const otherInfoDivs = sectionContent?.querySelectorAll('> *');
      const umpireDiv = otherInfoDivs?.find(otherInfoDiv => otherInfoDiv.innerHTML.includes('Umpires'));
      const umpire = getString(
         umpireDiv?.innerHTML.split('-')[1].split(',')[0].trim()
      ) as keyof typeof Umpire;
      if (!Object.keys(Umpire).includes(umpire)) {
         console.log(`No Umpire key for ${umpire}`);
         return false;
      }
      const weatherDiv = otherInfoDivs?.find(otherInfoDiv => otherInfoDiv.innerHTML.includes('Weather'));
      const temperature = Number(weatherDiv?.innerHTML.split('</strong>')[1].split('°')[0].trim());
      const scoreBox = dom.querySelector('.scorebox');
      const roadScoreBox = scoreBox?.querySelectorAll('> *')[0];
      const recordDiv = roadScoreBox?.querySelectorAll('> *')[2];
      const [wins, losses] = recordDiv?.innerText.split('-') as string[];
      const gameOfSeason = Number(wins) + Number(losses);
      const gameData: GameData = {
         baseballReferenceId,
         dayOfYear,
         gameOfSeason,
         hourOfDay,
         playingSurface: PlayingSurface[playingSurface],
         temperature,
         umpire: Umpire[umpire],
         venue: Venue[venue],
      };
      return gameData;
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
      console.log(baseballReferenceIds);
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
   const { html, season, url } = boxscores[0];
   if (!html)
      return;
   const dom = parse(html);
   const gameData = extractGameData(dom, url);
   if (gameData === false)
      return;
   console.log(gameData);
   const playerData = await extractPlayerData(dom);
}