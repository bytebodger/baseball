import dayjs from 'dayjs';
import * as https from 'https';
import { parse } from 'node-html-parser';
import type { WebSchedule } from '../interfaces/tables/WebSchedule.js';
import { getWebSchedules } from './queries/getWebSchedules.js';
import { insertWebSchedule } from './queries/insertWebSchedule.js';
import { updateWebSchedule } from './queries/updateWebSchedule.js';

export const scrape = () => {
   (async () => {
      const { rowCount, rows: webSchedule } = await getWebSchedules() as { rowCount: number, rows: WebSchedule[] };
      let targetSeason = Number(process.env.FIRST_YEAR);
      let targetSeasonIsNew = true;
      let targetSeasonWasProcessed = false;
      let webScheduleId = 0;
      if (rowCount) {
         const currentYear = dayjs().utc().year();
         const { is_complete, season, web_schedule_id } = webSchedule[0];
         if (!is_complete) {
            targetSeason = season;
            targetSeasonIsNew = false;
            webScheduleId = web_schedule_id;
         } else {
            targetSeason++;
            if (targetSeason > currentYear)
               return;
         }
      }
      const url = `https://www.baseball-reference.com/leagues/majors/${targetSeason}-schedule.shtml`;
      //const url = 'https://www.foo.com';
      https.get(url, response => {
         let html = '';
         response.on('data', chunk => html += chunk);
         response.on('end', async () => {
            let allGamesScraped = false;
            const doc = parse(html);
            const mlbSchedule = doc.querySelector('span[data-label="MLB Schedule"]')?.parentNode.parentNode;
            const sectionContent = mlbSchedule?.querySelector('.section_content');
            const days = sectionContent?.querySelectorAll('> *');
            days?.forEach(day => {
               const dayDescription = day.querySelector('h3');
               console.log(dayDescription?.innerText);
               const games = day.querySelector('.game');
            })
            if (targetSeasonIsNew) {
               console.log('inserting');
               await insertWebSchedule(
                  url,
                  html,
                  dayjs().utc().valueOf(),
                  targetSeason,
                  targetSeasonWasProcessed,
               )
            } else {
               console.log('updating');
               await updateWebSchedule(
                  webScheduleId,
                  html,
                  dayjs().utc().valueOf(),
                  targetSeasonWasProcessed,
               )
            }
         });
      });
   })()
}