import dayjs from 'dayjs';
import { parse } from 'node-html-parser';
import { page } from '../constants/page.js';
import type { Result } from '../interfaces/Result.js';
import type { WebBoxscoreTable } from '../interfaces/tables/WebBoxscoreTable.js';
import type { WebScheduleTable } from '../interfaces/tables/WebScheduleTable.js';
import { output } from './output.js';
import { getWebBoxscores } from './queries/getWebBoxscores.js';
import { getWebSchedules } from './queries/getWebSchedules.js';
import { insertWebBoxscore } from './queries/insertWebBoxscore.js';
import { insertWebSchedule } from './queries/insertWebSchedule.js';
import { updateWebSchedule } from './queries/updateWebSchedule.js';

export const retrieveWebSchedules = async () => {
   const result: Result = {
      errors: [],
      function: 'retrieveWebSchedules()',
      messages: [],
      proceed: false,
   };
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
         if (targetSeason > currentYear) {
            result.messages.push('All finished seasons have been processed.');
            return output(result);
         }
      }
   }
   const url = `https://www.baseball-reference.com/leagues/majors/${targetSeason}-schedule.shtml`;
   if (!targetSeasonIsNew && (hasBeenPlayed || lastChecked.isBefore(oneHourAgo))) {
      result.messages.push('This season was already checked within the last hour.');
      return output(result);
   }
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
            await insertWebBoxscore({
               season: targetSeason,
               url: boxscoreUrl,
            });
            result.messages.push('inserted web boxscore:');
            result.messages.push({
               season: targetSeason,
               url: boxscoreUrl,
            })
         }
      })
   })
   const now = dayjs().utc().unix();
   if (targetSeasonIsNew) {
      await insertWebSchedule({
         has_been_played: allGamesHaveBeenPlayed,
         html,
         season: targetSeason,
         time_checked: now,
         time_processed: allGamesHaveBeenPlayed ? now : null,
         time_retrieved: now,
         url,
      })
      result.messages.push('inserted web schedule:');
      result.messages.push({
         has_been_played: allGamesHaveBeenPlayed,
         season: targetSeason,
         time_checked: now,
         time_processed: allGamesHaveBeenPlayed ? now : null,
         time_retrieved: now,
         url,
      });
   } else {
      if (targetSeason === thisSeason && hasBeenPlayed) {
         result.messages.push('The current season has been completed and processed');
         return output(result);
      }
      await updateWebSchedule({
         has_been_played: allGamesHaveBeenPlayed,
         html,
         time_checked: now,
         time_processed: allGamesHaveBeenPlayed ? now : null,
         time_retrieved: now,
         web_schedule_id: webScheduleId,
      })
      result.messages.push('updated web schedule:');
      result.messages.push({
         has_been_played: allGamesHaveBeenPlayed,
         time_checked: now,
         time_processed: allGamesHaveBeenPlayed ? now : null,
         time_retrieved: now,
         web_schedule_id: webScheduleId,
      });
   }
   result.proceed = true;
   return output(result);
}