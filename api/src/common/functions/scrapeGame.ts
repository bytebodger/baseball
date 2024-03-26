import type { Dayjs } from 'dayjs';
import dayjs from 'dayjs';
import type { HTMLElement } from 'node-html-parser';
import { coversTeam } from '../constants/coversTeam.js';
import { PlayingSurface } from '../enums/PlayingSurface.js';
import { Team } from '../enums/Team.js';
import { Venue } from '../enums/Venue.js';
import type { Result } from '../interfaces/Result.js';
import type { GameTable } from '../interfaces/tables/GameTable.js';
import type { HistoricalOddsTable } from '../interfaces/tables/HistoricalOddsTable.js';
import type { TeamTable } from '../interfaces/tables/TeamTable.js';
import type { UmpireTable } from '../interfaces/tables/UmpireTable.js';
import { getString } from './getString.js';
import { getDBGame } from './queries/getDBGame.js';
import { getDBHistoricalOdds } from './queries/getDBHistoricalOdds.js';
import { getDBTeam } from './queries/getDBTeam.js';
import { getDBUmpire } from './queries/getDBUmpire.js';
import { insertDBGame } from './queries/insertDBGame.js';
import { insertDBUmpire } from './queries/insertDBUmpire.js';
import { removeDiacritics } from './removeDiacritics.js';
import { retrieveCoversOdds } from './retrieveCoversOdds.js';

export const scrapeGame = async (baseballReferenceId: string, dom: HTMLElement, result: Result) => {
   interface GameDay {
      dayjs: Dayjs,
      dayOfMonth: number,
      dayOfMonthString: string,
      dayOfYear: number,
      hourOfDay: number,
      month: number,
   }

   const getDoubleHeader = () => {
      const metaDivs = dom.querySelectorAll('.scorebox_meta > *');
      const game1 = metaDivs.some(
         metaDiv => metaDiv.innerHTML.includes('First game of doubleheader')
      );
      const game2 = metaDivs.some(
         metaDiv => metaDiv.innerHTML.includes('Second game of doubleheader')
      );
      return {
         game1,
         game2,
      }
   }

   const getGameDay = () => {
      const metaDivs = dom.querySelectorAll('.scorebox_meta > *');
      const gameDayString = metaDivs[0].innerText.split(',').slice(1).join(',').trim();
      const gameDay = dayjs(gameDayString).utc(true);
      const month = gameDay.month() + 1;
      const dayOfMonth = gameDay.date();
      const dayOfMonthString = dayOfMonth < 10 ? `0${dayOfMonth}` : dayOfMonth.toString();
      const dayOfYear = gameDay.dayOfYear();
      const startTimeDiv = metaDivs.find(metaDiv => metaDiv.innerText.includes('Start Time:'));
      let hourOfDay;
      if (!startTimeDiv) {
         hourOfDay = 19;
      } else {
         const [time, amPm] = startTimeDiv.innerText.split(':').slice(1).join(':').trim().split(' ').slice(0, 2);
         hourOfDay = Number(time.split(':').shift());
         if (amPm === 'a.m.' && hourOfDay === 12)
            hourOfDay = 24;
         else if (amPm === 'p.m.' && hourOfDay < 12)
            hourOfDay += 12;
      }
      return {
         dayjs: gameDay,
         dayOfMonth,
         dayOfMonthString,
         dayOfYear,
         hourOfDay,
         month,
      }
   }

   const getGameOfSeason = () => {
      const scoreBox = dom.querySelector('.scorebox');
      const roadScoreBox = scoreBox?.querySelectorAll('> *')[0];
      const recordDiv = roadScoreBox?.querySelectorAll('> *')[2];
      const [wins, losses] = recordDiv?.innerText.split('-') as string[];
      return Number(wins) + Number(losses);
   }

   const getHostScore = () => {
      const scoreboxDiv = dom.querySelector('.scorebox');
      if (!scoreboxDiv) {
         result.errors.push('No score box div while getting host score');
         return false;
      }
      const scoreboxSubDivs = scoreboxDiv.querySelectorAll('> *');
      const hostDiv = scoreboxSubDivs[1];
      const hostSubDivs = hostDiv.querySelectorAll('> *');
      return Number(hostSubDivs[1].querySelector('.score')?.innerText);
   }

   const getHostTeamId = async (hostTeamKey: keyof typeof Team) => {
      const { rows: host } = await getDBTeam(Team[hostTeamKey]) as { rows: TeamTable[] };
      if (host.length === 0) {
         result.errors.push(`No team ID found for ${hostTeamKey}`);
         return false;
      }
      return host[0].team_id;
   }

   const getHostTeamKey = () => {
      const scoreboxDiv = dom.querySelector('.scorebox');
      if (!scoreboxDiv) {
         result.errors.push('No score box div while getting host team key');
         return false;
      }
      const scoreboxSubDivs = scoreboxDiv.querySelectorAll('> *');
      const hostDiv = scoreboxSubDivs[1];
      const hostSubDivs = hostDiv.querySelectorAll('> *');
      const hostStrong = hostSubDivs[0].querySelector('strong');
      if (!hostStrong) {
         result.errors.push('No strong tag while getting host team key');
         return false;
      }
      const hostA = hostStrong.querySelector('a');
      const hostAHref = hostA?.getAttribute('href');
      const hostTeamKey = getString(hostAHref?.split('/')[2]) as keyof typeof Team;
      if (!Object.keys(Team).includes(hostTeamKey)) {
         result.errors.push(`No Team key for host: ${hostTeamKey}`);
         return false;
      }
      return hostTeamKey;
   }

   const getOdds = async (season: number, visitorTeamKey: keyof typeof Team, hostTeamKey: keyof typeof Team, gameDay: GameDay) => {
      let hostMoneyline = null;
      let overMoneyline = null;
      let overUnder = null;
      let underMoneyline = null;
      let visitorMoneyline = null;
      let odds: HistoricalOddsTable | null = null;
      if (season >= 2010 && season <= 2021) {
         const date = `${gameDay.month}${gameDay.dayOfMonthString}`;
         const { rows: historicalOdds } = await getDBHistoricalOdds(
            season,
            date,
            Team[visitorTeamKey],
            Team[hostTeamKey],
         ) as { rows: HistoricalOddsTable[] };
         if (historicalOdds.length) {
            if (historicalOdds.length === 1) {
               odds = historicalOdds[0];
            } else {
               if (doubleHeader.game1)
                  odds = historicalOdds[0];
               else if (doubleHeader.game2)
                  odds = historicalOdds[1];
            }
            if (odds) {
               hostMoneyline = odds.host_moneyline;
               overMoneyline = odds.over_moneyline;
               overUnder = odds.over_under;
               underMoneyline = odds.under_moneyline;
               visitorMoneyline = odds.visitor_moneyline;
            }
         }
      } else if (season >= 2022) {
         const dateString = gameDay.dayjs.format('YYYY-M-D');
         const coversOdds = await retrieveCoversOdds(
            dateString,
            coversTeam[visitorTeamKey],
            coversTeam[hostTeamKey],
         );
         if (coversOdds !== false) {
            hostMoneyline = coversOdds.hostMoneyline;
            overMoneyline = coversOdds.overMoneyline;
            overUnder = coversOdds.overUnder;
            underMoneyline = coversOdds.underMoneyline;
            visitorMoneyline = coversOdds.visitorMoneyline;
         }
      }
      return {
         hostMoneyline,
         overMoneyline,
         overUnder,
         underMoneyline,
         visitorMoneyline,
      }
   }

   const getPlayingSurface = () => {
      const metaDivs = dom.querySelectorAll('.scorebox_meta > *');
      const surfaceDiv = metaDivs.find(metaDiv => metaDiv.innerHTML.includes(', on '));
      const playingSurface = getString(surfaceDiv?.innerHTML.split(', on ').pop()) as keyof typeof PlayingSurface;
      if (!Object.keys(PlayingSurface).includes(playingSurface)) {
         result.errors.push(`No PlayingSurface key for ${playingSurface}`);
         return false;
      }
      return PlayingSurface[playingSurface];
   }

   const getSeason = (baseballReferenceId: string) => Number(baseballReferenceId.substring(7, 11));

   const getTemperature = () => {
      const otherInfo = dom.querySelector('span[data-label="Other Info"]')?.parentNode.parentNode;
      const sectionContent = otherInfo?.querySelector('.section_content');
      const otherInfoDivs = sectionContent?.querySelectorAll('> *');
      const weatherDiv = otherInfoDivs?.find(otherInfoDiv => otherInfoDiv.innerHTML.includes('Weather'));
      return Number(weatherDiv?.innerHTML.split('</strong>')[1].split('°')[0].trim());
   }

   const getUmpireId = async () => {
      const otherInfo = dom.querySelector('span[data-label="Other Info"]')?.parentNode.parentNode;
      const sectionContent = otherInfo?.querySelector('.section_content');
      const otherInfoDivs = sectionContent?.querySelectorAll('> *');
      const umpireDiv = otherInfoDivs?.find(otherInfoDiv => otherInfoDiv.innerHTML.includes('Umpires'));
      const name = removeDiacritics(getString(umpireDiv?.innerHTML.split('-')[1].split(',')[0].trim()));
      const { rows: umpire } = await getDBUmpire(name) as { rows: UmpireTable[] };
      if (umpire.length)
         return umpire[0].umpire_id;
      const { rows: newUmpire } = await insertDBUmpire({ name }) as { rows: UmpireTable[] };
      return newUmpire[0].umpire_id;
   }

   const getVenue = () => {
      const metaDivs = dom.querySelectorAll('.scorebox_meta > *');
      const venueDiv = metaDivs.find(metaDiv => metaDiv.innerHTML.includes('Venue'));
      const venue = getString(
         venueDiv?.innerHTML.split(':').pop()?.trim().replace('"', '').replace('amp;', '')
      ) as keyof typeof Venue;
      if (!Object.keys(Venue).includes(venue)) {
         result.errors.push(`No Venue key for ${venue}`);
         return false;
      }
      return Venue[venue];
   }

   const getVisitorScore = () => {
      const scoreboxDiv = dom.querySelector('.scorebox');
      if (!scoreboxDiv) {
         result.errors.push('No score box div while getting visitor score');
         return false;
      }
      const scoreboxSubDivs = scoreboxDiv.querySelectorAll('> *');
      const visitorDiv = scoreboxSubDivs[0];
      const visitorSubDivs = visitorDiv.querySelectorAll('> *');
      return Number(visitorSubDivs[1].querySelector('.score')?.innerText);
   }

   const getVisitorTeamId = async (visitorTeamKey: keyof typeof Team) => {
      const { rows: visitor } = await getDBTeam(Team[visitorTeamKey]) as { rows: TeamTable[] };
      if (visitor.length === 0) {
         result.errors.push(`No team ID found for ${visitorTeamKey}`);
         return false;
      }
      return visitor[0].team_id;
   }

   const getVisitorTeamKey = () => {
      const scoreboxDiv = dom.querySelector('.scorebox');
      if (!scoreboxDiv) {
         result.errors.push('No score box div while getting visitor team key');
         return false;
      }
      const scoreboxSubDivs = scoreboxDiv.querySelectorAll('> *');
      const visitorDiv = scoreboxSubDivs[0];
      const visitorSubDivs = visitorDiv.querySelectorAll('> *');
      const visitorStrong = visitorSubDivs[0].querySelector('strong');
      if (!visitorStrong) {
         result.errors.push('No strong tag while getting visitor team key');
         return false;
      }
      const visitorA = visitorStrong.querySelector('a');
      const visitorAHref = visitorA?.getAttribute('href');
      const visitorTeamKey = getString(visitorAHref?.split('/')[2]) as keyof typeof Team;
      if (!Object.keys(Team).includes(visitorTeamKey)) {
         result.errors.push(`No Team key for visitor: ${visitorTeamKey}`);
         return false;
      }
      return visitorTeamKey;
   }

   const { rows: game } = await getDBGame(baseballReferenceId) as { rows: GameTable[] };
   if (game.length)
      return game[0];
   const season = getSeason(baseballReferenceId);
   const visitorTeamKey = getVisitorTeamKey();
   if (result.errors.length || visitorTeamKey === false)
      return false;
   const visitorTeamId = await getVisitorTeamId(visitorTeamKey);
   if (result.errors.length || visitorTeamId === false)
      return false;
   const visitorScore = getVisitorScore();
   if (result.errors.length || visitorScore === false)
      return false;
   const hostTeamKey = getHostTeamKey();
   if (result.errors.length || hostTeamKey === false)
      return false;
   const hostTeamId = await getHostTeamId(hostTeamKey);
   if (result.errors.length || hostTeamId === false)
      return false;
   const hostScore = getHostScore();
   if (result.errors.length || hostScore === false)
      return false;
   const gameDay = getGameDay();
   if (result.errors.length)
      return false;
   const venue = getVenue();
   if (result.errors.length || venue === false)
      return false;
   const playingSurface = getPlayingSurface();
   if (result.errors.length || playingSurface === false)
      return false;
   const doubleHeader = getDoubleHeader();
   const umpireId = await getUmpireId();
   const temperature = getTemperature();
   const gameOfSeason = getGameOfSeason();
   const odds = await getOdds(
      season,
      visitorTeamKey,
      hostTeamKey,
      gameDay,
   );
   const fields = {
      baseball_reference_id: baseballReferenceId,
      day_of_year: gameDay.dayOfYear,
      game_of_season: gameOfSeason,
      home_plate_umpire: umpireId,
      host_moneyline: odds.hostMoneyline,
      host_score: hostScore,
      host_team_id: hostTeamId,
      hour_of_day: gameDay.hourOfDay,
      over_moneyline: odds.overMoneyline,
      over_under: odds.overUnder,
      playing_surface: playingSurface,
      season,
      temperature,
      under_moneyline: odds.underMoneyline,
      venue,
      visitor_moneyline: odds.visitorMoneyline,
      visitor_score: visitorScore,
      visitor_team_id: visitorTeamId,
   }
   const { rows: newGame } = await insertDBGame(fields) as { rows: GameTable[] };
   result.messages.push('inserted game:');
   result.messages.push(fields);
   return newGame[0];
}