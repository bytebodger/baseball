import dayjs from 'dayjs';
import * as https from 'https';
import { parse } from 'node-html-parser';
import { WebBoxscore } from '../interfaces/tables/WebBoxscore.js';
import type { WebSchedule } from '../interfaces/tables/WebSchedule.js';
import { getWebBoxscores } from './queries/getWebBoxscores.js';
import { getWebSchedules } from './queries/getWebSchedules.js';
import { insertWebBoxscore } from './queries/insertWebBoxscore.js';
import { insertWebSchedule } from './queries/insertWebSchedule.js';
import { updateWebSchedule } from './queries/updateWebSchedule.js';

export const scrapeWebSchedules = () => {
   (async () => {
      const { rows: webBoxScores } = await getWebBoxscores() as { rows: WebBoxscore[] };
      const {
         rowCount: webSchedulesRowCount,
         rows: webSchedules,
      } = await getWebSchedules() as { rowCount: number, rows: WebSchedule[] };
      const { is_complete, season, web_schedule_id } = webSchedules[0];
      let targetSeason = Number(process.env.FIRST_YEAR);
      let targetSeasonIsNew = true;
      let webScheduleId = 0;
      if (webSchedulesRowCount) {
         const currentYear = dayjs().utc().year();
         if (!is_complete) {
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
      if (!targetSeasonIsNew && is_complete)
         return;
      https.get(url, response => {
         let html = '';
         response.on('data', chunk => html += chunk);
         response.on('end', async () => {
            let allGamesCompleted = true;
            const doc = parse(html);
            const mlbSchedule = doc.querySelector('span[data-label="MLB Schedule"]')?.parentNode.parentNode;
            const sectionContent = mlbSchedule?.querySelector('.section_content');
            const days = sectionContent?.querySelectorAll('> *');
            days?.map(async day => {
               if (!allGamesCompleted)
                  return;
               const games = day.querySelectorAll('.game');
               games.map(async game => {
                  if (!allGamesCompleted)
                     return;
                  const spans = game.querySelectorAll('span');
                  if (spans.some(span => span.innerText === '(Spring)'))
                     return;
                  const em = game.querySelector('em');
                  if (!em)
                     allGamesCompleted = false;
                  const a = em?.querySelector('a');
                  if (!a?.getAttribute('href'))
                     return;
                  const boxscoreUrl = 'https://baseball-reference.com' + a?.getAttribute('href');
                  if (!webBoxScores.some(webBoxScore => webBoxScore.url === boxscoreUrl))
                     await insertWebBoxscore(boxscoreUrl, targetSeason);
               })
            })
            if (targetSeasonIsNew) {
               await insertWebSchedule(
                  url,
                  html,
                  dayjs().utc().valueOf(),
                  targetSeason,
                  allGamesCompleted,
               )
            } else {
               await updateWebSchedule(
                  webScheduleId,
                  html,
                  dayjs().utc().valueOf(),
                  allGamesCompleted,
               )
            }
         });
      });
   })()
}