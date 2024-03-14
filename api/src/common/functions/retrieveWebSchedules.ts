import dayjs from 'dayjs';
import { parse } from 'node-html-parser';
import type { Page } from 'puppeteer';
import { Milliseconds } from '../enums/Milliseconds.js';
import type { WebBoxscore } from '../interfaces/tables/WebBoxscore.js';
import type { WebSchedule } from '../interfaces/tables/WebSchedule.js';
import { getWebBoxscores } from './queries/getWebBoxscores.js';
import { getWebSchedules } from './queries/getWebSchedules.js';
import { insertWebBoxscore } from './queries/insertWebBoxscore.js';
import { insertWebSchedule } from './queries/insertWebSchedule.js';
import { updateWebSchedule } from './queries/updateWebSchedule.js';
import { sleep } from './sleep.js';

export const retrieveWebSchedules = async (page: Page): Promise<void> => {
   const { rows: webBoxscores } = await getWebBoxscores() as { rows: WebBoxscore[] };
   const { rows: webSchedules, } = await getWebSchedules() as { rows: WebSchedule[] };
   let hasBeenPlayed = false;
   let targetSeason = Number(process.env.FIRST_YEAR);
   let thisSeason = Number(process.env.FIRST_YEAR);
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
   const mlbSchedule = dom.querySelector('span[data-label="MLB Schedule"]')?.parentNode.parentNode;
   const sectionContent = mlbSchedule?.querySelector('.section_content');
   const days = sectionContent?.querySelectorAll('> *');
   days?.map(async day => {
      if (!allGamesHaveBeenPlayed)
         return;
      const games = day.querySelectorAll('.game');
      games.map(async game => {
         if (!allGamesHaveBeenPlayed)
            return;
         const spans = game.querySelectorAll('span');
         if (spans.some(span => span.innerText === '(Spring)'))
            return;
         const em = game.querySelector('em');
         if (!em)
            allGamesHaveBeenPlayed = false;
         const a = em?.querySelector('a');
         if (!a?.getAttribute('href'))
            return;
         const boxscoreUrl = 'https://www.baseball-reference.com' + a.getAttribute('href');
         if (!webBoxscores.some(webBoxScore => webBoxScore.url === boxscoreUrl))
            await insertWebBoxscore({
               season: targetSeason,
               url: boxscoreUrl,
            });
      })
   })
   const now = dayjs().utc().valueOf();
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
   } else {
      if (targetSeason === thisSeason && hasBeenPlayed)
         return;
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