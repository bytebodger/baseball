import dayjs from 'dayjs';
import { parse } from 'node-html-parser';
import type { Page } from 'puppeteer';
import { Milliseconds } from '../enums/Milliseconds.js';
import type { WebBoxscoreTable } from '../interfaces/tables/WebBoxscoreTable.js';
import type { WebScheduleTable } from '../interfaces/tables/WebScheduleTable.js';
import { getWebBoxscores } from './queries/getWebBoxscores.js';
import { getWebSchedules } from './queries/getWebSchedules.js';
import { insertWebBoxscore } from './queries/insertWebBoxscore.js';
import { insertWebSchedule } from './queries/insertWebSchedule.js';
import { updateWebSchedule } from './queries/updateWebSchedule.js';
import { sleep } from './sleep.js';

export const retrieveWebSchedules = async (page: Page): Promise<void> => {
   const { rows: webBoxscores } = await getWebBoxscores() as { rows: WebBoxscoreTable[] };
   const { rows: webSchedules, } = await getWebSchedules() as { rows: WebScheduleTable[] };
   const earliestSeason = 2020;
   let hasBeenPlayed = false;
   let targetSeason = earliestSeason;
   let thisSeason = earliestSeason;
   let targetSeasonIsNew = true;
   let lastChecked = dayjs().utc();
   let webScheduleId = 0;
   const oneHourAgo = dayjs().utc().subtract(1, 'hour');
   if (webSchedules.length) {
      const { has_been_played, season, time_checked, web_schedule_id } = webSchedules[0];
      thisSeason = season;
      hasBeenPlayed = has_been_played;
      lastChecked = dayjs(time_checked).utc(true);
      const currentYear = dayjs().utc().year();
      if (!hasBeenPlayed) {
         targetSeason = season;
         targetSeasonIsNew = false;
         webScheduleId = web_schedule_id;
      } else {
         targetSeason = season + 1;
         if (targetSeason > currentYear)
            return;
      }
   }
   const url = `https://www.baseball-reference.com/leagues/majors/${targetSeason}-schedule.shtml`;
   if (!targetSeasonIsNew && (hasBeenPlayed || lastChecked.isBefore(oneHourAgo)))
      return;
   await page.goto(url, { waitUntil: 'domcontentloaded' });
   const html = await page.content();
   let allGamesHaveBeenPlayed = true;
   const dom = parse(html);
   const days = dom.querySelector('span[data-label="MLB Schedule"]')
      ?.parentNode
      .parentNode
      .querySelectorAll('.section_content > *');
   days?.map(async day => {
      if (!allGamesHaveBeenPlayed)
         return;
      const games = day.querySelectorAll('.game');
      games.map(async game => {
         if (!allGamesHaveBeenPlayed)
            return;
         const emA = game.querySelector('em a');
         if (!emA) {
            allGamesHaveBeenPlayed = false;
            return;
         }
         if (game.querySelectorAll('span').length)
            return;
         const href = emA?.getAttribute('href');
         if (!href)
            return;
         const boxscoreUrl = `https://www.baseball-reference.com${href}`;
         if (!webBoxscores.some(webBoxScore => webBoxScore.url === boxscoreUrl)) {
            console.log('inserting web boxscore', {
               season: targetSeason,
               url: boxscoreUrl,
            })
            await insertWebBoxscore({
               season: targetSeason,
               url: boxscoreUrl,
            });
         }
      })
   })
   const now = dayjs().utc().unix();
   if (targetSeasonIsNew) {
      console.log('inserting web schedule', {
         has_been_played: allGamesHaveBeenPlayed,
         season: targetSeason,
         time_checked: now,
         time_processed: allGamesHaveBeenPlayed ? now : null,
         time_retrieved: now,
         url,
      })
      await insertWebSchedule({
         has_been_played: allGamesHaveBeenPlayed,
         html,
         season: targetSeason,
         time_checked: now,
         time_processed: allGamesHaveBeenPlayed ? now : null,
         time_retrieved: now,
         url,
      })
   } else {
      if (targetSeason === thisSeason && hasBeenPlayed)
         return;
      console.log('updating web schedule', {
         has_been_played: allGamesHaveBeenPlayed,
         time_checked: now,
         time_processed: allGamesHaveBeenPlayed ? now : null,
         time_retrieved: now,
         web_schedule_id: webScheduleId,
      })
      await updateWebSchedule({
         has_been_played: allGamesHaveBeenPlayed,
         html,
         time_checked: now,
         time_processed: allGamesHaveBeenPlayed ? now : null,
         time_retrieved: now,
         web_schedule_id: webScheduleId,
      })
   }
   await sleep(4 * Milliseconds.second);
   return await retrieveWebSchedules(page);
}