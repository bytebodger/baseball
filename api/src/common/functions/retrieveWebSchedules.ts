import dayjs from 'dayjs';
import { parse } from 'node-html-parser';
import { Milliseconds } from '../enums/Milliseconds.js';
import { WebBoxscore } from '../interfaces/tables/WebBoxscore.js';
import type { WebSchedule } from '../interfaces/tables/WebSchedule.js';
import { getHttps } from './getHttps.js';
import { getWebBoxscores } from './queries/getWebBoxscores.js';
import { getWebSchedules } from './queries/getWebSchedules.js';
import { insertWebBoxscore } from './queries/insertWebBoxscore.js';
import { insertWebSchedule } from './queries/insertWebSchedule.js';
import { updateWebSchedule } from './queries/updateWebSchedule.js';

export const retrieveWebSchedules = async () => {
   const { rows: webBoxScores } = await getWebBoxscores() as { rows: WebBoxscore[] };
   const { rows: webSchedules, } = await getWebSchedules() as { rows: WebSchedule[] };
   let hasBeenPlayed = false;
   let targetSeason = Number(process.env.FIRST_YEAR);
   let thisSeason = Number(process.env.FIRST_YEAR);
   let targetSeasonIsNew = true;
   let webScheduleId = 0;
   if (webSchedules.length) {
      const { has_been_played, season, web_schedule_id } = webSchedules[0];
      thisSeason = season;
      hasBeenPlayed = has_been_played;
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
   if (!targetSeasonIsNew && hasBeenPlayed)
      return;
   const html = await getHttps(url);
   let allGamesHaveBeenPlayed = true;
   const doc = parse(html);
   const mlbSchedule = doc.querySelector('span[data-label="MLB Schedule"]')?.parentNode.parentNode;
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
         if (!webBoxScores.some(webBoxScore => webBoxScore.url === boxscoreUrl))
            await insertWebBoxscore(boxscoreUrl, targetSeason);
      })
   })
   const now = dayjs().utc().valueOf();
   if (targetSeasonIsNew) {
      await insertWebSchedule(
         allGamesHaveBeenPlayed,
         html,
         targetSeason,
         allGamesHaveBeenPlayed ? now : null,
         now,
         url,
      )
   } else {
      if (targetSeason === thisSeason && hasBeenPlayed)
         return;
      await updateWebSchedule({
         has_been_played: allGamesHaveBeenPlayed,
         html,
         time_processed: allGamesHaveBeenPlayed ? now : null,
         time_retrieved: now,
         web_scheduled_id: webScheduleId,
      })
   }
   setTimeout(() => retrieveWebSchedules(), 4 * Milliseconds.second);
}