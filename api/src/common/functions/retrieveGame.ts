import dayjs from 'dayjs';
import type { HTMLElement } from 'node-html-parser';
import { PlayingSurface } from '../enums/PlayingSurface.js';
import { Team } from '../enums/Team.js';
import { Umpire } from '../enums/Umpire.js';
import { Venue } from '../enums/Venue.js';
import type { Game } from '../interfaces/tables/Game.js';
import type { HistoricalOdds } from '../interfaces/tables/HistoricalOdds.js';
import type { Team as TeamTable } from '../interfaces/tables/Team.js';
import { getString } from './getString.js';
import { getGame } from './queries/getGame.js';
import { getHistoricalOdds } from './queries/getHistoricalOdds.js';
import { getTeam } from './queries/getTeam.js';
import { insertGame } from './queries/insertGame.js';

export const retrieveGame = async (baseballReferenceId: string, dom: HTMLElement) => {
   const { rows: game } = await getGame(baseballReferenceId) as { rows: Game[] };
   if (game.length)
      return game[0];
   const season = Number(baseballReferenceId.substring(7, 11));
   const scoreboxDiv = dom.querySelector('.scorebox');
   if (!scoreboxDiv)
      return false;
   const scoreboxSubDivs = scoreboxDiv.querySelectorAll('> *');
   const [visitorDiv, hostDiv] = scoreboxSubDivs;
   const visitorSubDivs = visitorDiv.querySelectorAll('> *');
   const visitorStrong = visitorSubDivs[0].querySelector('strong');
   if (!visitorStrong)
      return false;
   const visitorA = visitorStrong.querySelector('a');
   const visitorAHref = visitorA?.getAttribute('href');
   const visitorAbbreviation = getString(visitorAHref?.split('/')[2]) as keyof typeof Team;
   if (!Object.keys(Team).includes(visitorAbbreviation)) {
      console.log(`No Team key for visitor: ${visitorAbbreviation}`);
      return false;
   }
   const { rows: visitor } = await getTeam(Team[visitorAbbreviation]) as { rows: TeamTable[] };
   const visitorScore = Number(visitorSubDivs[1].querySelector('.score')?.innerText);
   const hostSubDivs = hostDiv.querySelectorAll('> *');
   const hostStrong = hostSubDivs[0].querySelector('strong');
   if (!hostStrong)
      return false;
   const hostA = hostStrong.querySelector('a');
   const hostAHref = hostA?.getAttribute('href');
   const hostAbbreviation = getString(hostAHref?.split('/')[2]) as keyof typeof Team;
   if (!Object.keys(Team).includes(hostAbbreviation)) {
      console.log(`No Team key for host: ${hostAbbreviation}`);
      return false;
   }
   const { rows: host } = await getTeam(Team[hostAbbreviation]) as { rows: TeamTable[] };
   const hostScore = Number(hostSubDivs[1].querySelector('.score')?.innerText);
   const metaDivs = dom.querySelectorAll('.scorebox_meta > *');
   const gameDayString = metaDivs[0].innerText.split(',').slice(1).join(',').trim();
   const gameDay = dayjs(gameDayString).utc(true);
   const month = gameDay.month() + 1;
   const dayOfMonth = gameDay.date();
   const dayOfMonthString = dayOfMonth < 10 ? `0${dayOfMonth}` : dayOfMonth.toString();
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
   const isDoubleheaderGame1 = metaDivs.find(
      metaDiv => metaDiv.innerHTML.includes('First game of doubleheader')
   );
   const isDoubleheaderGame2 = metaDivs.find(
      metaDiv => metaDiv.innerHTML.includes('Second game of doubleheader')
   );
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
   let hostMoneyline = null;
   let overMoneyline = null;
   let overUnder = null;
   let underMoneyline = null;
   let visitorMoneyline = null;
   if (season >= 2010 && season <= 2021) {
      const date = `${month}${dayOfMonthString}`;
      const { rows: historicalOdds } = await getHistoricalOdds(
         season,
         date,
         Team[visitorAbbreviation],
         Team[hostAbbreviation],
      ) as { rows: HistoricalOdds[] };
      if (!historicalOdds.length)
         return;
      let odds: HistoricalOdds;
      if (historicalOdds.length === 1) {
         odds = historicalOdds[0];
      } else if (historicalOdds.length === 2) {
         if (isDoubleheaderGame1)
            odds = historicalOdds[0];
         else if (isDoubleheaderGame2)
            odds = historicalOdds[1];
         else
            return;
      } else
         return;
      hostMoneyline = odds.host_moneyline;
      overMoneyline = odds.over_moneyline;
      overUnder = odds.over_under;
      underMoneyline = odds.under_moneyline;
      visitorMoneyline = odds.visitor_moneyline;
   }
   const { rows: newGame } = await insertGame({
      baseball_reference_id: baseballReferenceId,
      day_of_year: dayOfYear,
      game_of_season: gameOfSeason,
      home_plate_umpire: Umpire[umpire],
      host_moneyline: hostMoneyline,
      host_score: hostScore,
      host_team_id: host[0].team_id,
      hour_of_day: hourOfDay,
      over_moneyline: overMoneyline,
      over_under: overUnder,
      playing_surface: PlayingSurface[playingSurface],
      season,
      temperature,
      under_moneyline: underMoneyline,
      venue: Venue[venue],
      visitor_moneyline: visitorMoneyline,
      visitor_score: visitorScore,
      visitor_team_id: visitor[0].team_id,
   }) as { rows: Game[] };
   return newGame[0];
}