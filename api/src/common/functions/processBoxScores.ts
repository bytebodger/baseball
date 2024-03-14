import dayjs from 'dayjs';
import { parse } from 'node-html-parser';
import { PlayingSurface } from '../enums/PlayingSurface.js';
import { Umpire } from '../enums/Umpire.js';
import { Venue } from '../enums/Venue.js';
import { WebBoxscore } from '../interfaces/tables/WebBoxscore.js';
import { getString } from './getString.js';
import { getOldestUnprocessedBoxscore } from './queries/getOldestUnprocessedBoxscore.js';

export const processBoxScores = async () => {
   const { rows: boxscores } = await getOldestUnprocessedBoxscore() as { rows: WebBoxscore[] };
   if (!boxscores.length)
      return Promise.resolve();
   const { html, season, url } = boxscores[0];
   console.log('url', url);
   if (!html)
      return Promise.resolve();
   const baseballReferenceId = getString(url.split('/').pop()).split('.').shift();
   console.log('baseballReferenceId', baseballReferenceId);
   const dom = parse(html);
   const metaDivs = dom.querySelectorAll('.scorebox_meta > *');
   const gameDayString = metaDivs[0].innerText.split(',').slice(1).join(',').trim();
   const gameDay = dayjs(gameDayString).utc(true);
   const dayOfYear = gameDay.dayOfYear();
   console.log('dayOfYear', dayOfYear);
   const [time, amPm] = metaDivs[1].innerText.split(':').slice(1).join(':').trim().split(' ').slice(0, 2);
   let hourOfDay = Number(time.split(':').shift());
   if (amPm === 'a.m.' && hourOfDay === 12)
      hourOfDay = 24;
   else if (amPm === 'p.m.' && hourOfDay < 12)
      hourOfDay+= 12;
   console.log('hourOfDay', hourOfDay);
   const venueDiv = metaDivs.find(metaDiv => metaDiv.innerHTML.includes('Venue'));
   const venue = getString(venueDiv?.innerHTML.split(':').pop()?.trim().replace('"', ''));
   console.log('venue', venue);
   if (!Object.keys(Venue).includes(venue)) {
      console.log(`No Venue key for ${venue}`);
      return Promise.resolve();
   }
   const surfaceDiv = metaDivs.find(metaDiv => metaDiv.innerHTML.includes(', on '));
   const playingSurface = getString(surfaceDiv?.innerHTML.split(', on ').pop());
   console.log('playingSurface', playingSurface);
   if (!Object.keys(PlayingSurface).includes(playingSurface)) {
      console.log(`No PlayingSurface key for ${playingSurface}`);
      return Promise.resolve();
   }
   const otherInfo = dom.querySelector('span[data-label="Other Info"]')?.parentNode.parentNode;
   const sectionContent = otherInfo?.querySelector('.section_content');
   const otherInfoDivs = sectionContent?.querySelectorAll('> *');
   const umpireDiv = otherInfoDivs?.find(otherInfoDiv => otherInfoDiv.innerHTML.includes('Umpires'));
   const umpire = getString(umpireDiv?.innerHTML.split('-')[1].split(',')[0].trim());
   console.log('umpire', umpire);
   if (!Object.keys(Umpire).includes(umpire)) {
      console.log(`No Umpire key for ${umpire}`);
      return Promise.resolve();
   }
   const weatherDiv = otherInfoDivs?.find(otherInfoDiv => otherInfoDiv.innerHTML.includes('Weather'));
   const temperature = Number(weatherDiv?.innerHTML.split('</strong>')[1].split('°')[0].trim());
   console.log('temperature', temperature);
   const scoreBox = dom.querySelector('.scorebox');
   const roadScoreBox = scoreBox?.querySelectorAll('> *')[0];
   const recordDiv = roadScoreBox?.querySelectorAll('> *')[2];
   const [wins, losses] = recordDiv?.innerText.split('-') as string[];
   const gameOfSeason = Number(wins) + Number(losses);
   console.log('gameOfSeason', gameOfSeason);
}