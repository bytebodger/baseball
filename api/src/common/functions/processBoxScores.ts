import dayjs from 'dayjs';
import type { HTMLElement } from 'node-html-parser';
import { parse } from 'node-html-parser';
import type { Page } from 'puppeteer';
import { AtBat } from '../enums/AtBat.js';
import { Pitch } from '../enums/Pitch.js';
import type { AtBatTable } from '../interfaces/tables/AtBatTable.js';
import type { GameTable } from '../interfaces/tables/GameTable.js';
import type { PlayerTable } from '../interfaces/tables/PlayerTable.js';
import type { WebBoxscoreTable } from '../interfaces/tables/WebBoxscoreTable.js';
import { getNumber } from './getNumber.js';
import { getString } from './getString.js';
import { getAtBats } from './queries/getAtBats.js';
import { getOldestUnprocessedBoxscore } from './queries/getOldestUnprocessedBoxscore.js';
import { insertAtBat } from './queries/insertAtBat.js';
import { insertPitch } from './queries/insertPitch.js';
import { updateWebBoxscore } from './queries/updateWebBoxscore.js';
import { removeDiacritics } from './removeDiacritics.js';
import { retrieveGame } from './retrieveGame.js';
import { retrievePlayer } from './retrievePlayer.js';

export const processBoxScores = async (page: Page): Promise<boolean> => {
   const extractAtBats = async (game: GameTable, players: PlayerTable[]) => {
      const { rows: atBats } = await getAtBats(game.game_id) as { rows: AtBatTable[] };
      let errorOccurred = false;
      const trs = dom.querySelectorAll('#play_by_play tbody > *');
      await Promise.all(trs.map(async tr => {
         if (errorOccurred)
            return;
         const trId = tr.getAttribute('id');
         if (!trId?.startsWith('event_'))
            return;
         const sequenceId = Number(trId?.split('_').pop());
         if (atBats.some(atBat => atBat.game_id === game.game_id && atBat.sequence_id === sequenceId))
            return;
         const tds = tr.querySelectorAll('td');
         const totalPitches = Number(tds[3].innerText.split(',').shift());
         const runs = getNumber(tds[4].innerText.match(/R/g)?.length);
         const outs = getNumber(tds[4].innerText.match(/O/g)?.length);
         const batterName = removeDiacritics(tds[6].innerText.replace(/&nbsp;/g, ' '));
         const batter = players.find(player => player.name === batterName);
         if (!batter) {
            console.log(`Could not find batter ${batterName} while getting at-bat`);
            errorOccurred = true;
            return;
         }
         const pitcherName = removeDiacritics(tds[7].innerText.replace(/&nbsp;/g, ' '));
         const pitcher = players.find(player => player.name === pitcherName);
         if (!pitcher) {
            console.log(`Could not find pitcher ${pitcherName} while getting at-bat`);
            errorOccurred = true;
            return;
         }
         const outcome = tds[10].innerText;
         let bases: number | null = null;
         if (
            outcome.includes('Double Play')
            || outcome.startsWith('Flyball')
            || outcome.startsWith('Foul Popfly')
            || outcome.startsWith('Groundout')
            || outcome.startsWith('Popfly')
            || outcome.startsWith('Lineout')
            || outcome.startsWith('Strikeout')
            || outcome.includes('Triple Play')
         )
            bases = 0;
         else if (
            outcome.startsWith('Hit By Pitch')
            || outcome.startsWith('Single to')
            || outcome.startsWith('Walk')
         )
            bases = 1;
         else if (
            outcome.startsWith('Double to')
            || outcome.startsWith('Ground-rule Double')
         )
            bases = 2;
         else if (outcome.startsWith('Triple to'))
            bases = 3;
         else if (outcome.startsWith('Home Run'))
            bases = 4;
         if (bases === null)
            return;
         let result: AtBat;
         if (outcome.includes('Double Play'))
            result = AtBat.doublePlay;
         else if (outcome.startsWith('Flyball'))
            result = AtBat.flyball;
         else if (outcome.startsWith('Foul Popfly'))
            result = AtBat.foulPopfly;
         else if (outcome.startsWith('Groundout'))
            result = AtBat.groundout;
         else if (outcome.startsWith('Popfly'))
            result = AtBat.popfly;
         else if (outcome.startsWith('Lineout'))
            result = AtBat.lineout;
         else if (outcome.startsWith('Strikeout'))
            result = AtBat.strikeout;
         else if (outcome.includes('Triple Play'))
            result = AtBat.triplePlay;
         else if (outcome.startsWith('Hit By Pitch'))
            result = AtBat.hitByPitch;
         else if (outcome.startsWith('Single to'))
            result = AtBat.single;
         else if (outcome.startsWith('Walk'))
            result = AtBat.walk;
         else if (outcome.startsWith('Double to'))
            result = AtBat.double;
         else if (outcome.startsWith('Ground-rule Double'))
            result = AtBat.groundRuleDouble;
         else if (outcome.startsWith('Triple to'))
            result = AtBat.triple;
         else if (outcome.startsWith('Home Run'))
            result = AtBat.homeRun;
         else
            result = AtBat.unknown;
         console.log('inserting at-bat', {
            bases,
            batter_player_id: batter.player_id,
            game_id: game.game_id,
            outs,
            pitcher_player_id: pitcher.player_id,
            result,
            runs,
            sequence_id: sequenceId,
            total_pitches: totalPitches,
         })
         const { rows: newAtBat } = await insertAtBat({
            bases,
            batter_player_id: batter.player_id,
            game_id: game.game_id,
            outs,
            pitcher_player_id: pitcher.player_id,
            result,
            runs,
            sequence_id: sequenceId,
            total_pitches: totalPitches,
         }) as { rows: AtBatTable[] }
         const pitches = getString(tds[3].querySelector('.pitch_sequence')?.innerText);
         [...pitches].map(async (pitch, index) => {
            let result: Pitch;
            switch (pitch) {
               case 'B':
                  result = Pitch.ball;
                  break;
               case 'C':
                  result = Pitch.calledStrike;
                  break;
               case 'F':
                  result = Pitch.foul;
                  break;
               case 'H':
                  result = Pitch.hitBatter;
                  break;
               case 'I':
                  result = Pitch.intentionalBall;
                  break;
               case 'K':
                  result = Pitch.unknownStrike;
                  break;
               case 'L':
                  result = Pitch.foulBunt;
                  break;
               case 'M':
                  result = Pitch.missedBuntAttempt;
                  break;
               case 'O':
                  result = Pitch.foulTipOnBunt;
                  break;
               case 'Q':
                  result = Pitch.swingingOnPitchout;
                  break;
               case 'R':
                  result = Pitch.foulBallOnPitchout;
                  break;
               case 'S':
                  result = Pitch.swingingStrike;
                  break;
               case 'T':
                  result = Pitch.foulTip;
                  break;
               case 'V':
                  result = Pitch.automaticIntentionalBall;
                  break;
               case 'X':
                  result = Pitch.ballPutIntoPlayByBatter;
                  break;
               case 'Y':
                  result = Pitch.ballPutIntoPlayOnPitchout;
                  break;
               default:
                  result = Pitch.unknownOrMissedPitch;
            }
            console.log('inserting pitch', {
               at_bat_id: newAtBat[0].at_bat_id,
               result,
               sequence_id: index + 1,
            })
            await insertPitch({
               at_bat_id: newAtBat[0].at_bat_id,
               result,
               sequence_id: index + 1,
            })
         })
      }))
      return !errorOccurred;
   }

   const extractGame = async (dom: HTMLElement, url: string) => {
      const baseballReferenceId = getString(url.split('/').slice(4).join('/').split('.').shift());
      return await retrieveGame(baseballReferenceId, dom, page);
   }

   const extractPlayers = async (dom: HTMLElement) => {
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
      return await retrievePlayers(baseballReferenceIds, []);
   }

   const retrievePlayers = async (baseballReferenceIds: string[], players: PlayerTable[]): Promise<PlayerTable[] | false> => {
      if (!baseballReferenceIds.length)
         return players;
      const baseballReferenceId = baseballReferenceIds.shift();
      if (!baseballReferenceId)
         return players;
      const player = await retrievePlayer(baseballReferenceId, page);
      if (player === false)
         return false;
      players.push(player);
      return await retrievePlayers(baseballReferenceIds, players);
   }

   const { rows: boxscores } = await getOldestUnprocessedBoxscore() as { rows: WebBoxscoreTable[] };
   if (!boxscores.length)
      return false;
   console.log('oldestUnprocessedBoxscore:');
   console.log(boxscores[0].web_boxscore_id);
   console.log(boxscores[0].url);
   console.log('');
   const { html, url } = boxscores[0];
   if (!html)
      return false;
   const dom = parse(html);
   const game = await extractGame(dom, url);
   if (game === false)
      return false;
   const players = await extractPlayers(dom);
   if (players === false)
      return false;
   const atBatResult = await extractAtBats(game, players);
   if (!atBatResult)
      return false;
   console.log('updating web boxscore', {
      web_boxscore_id: boxscores[0].web_boxscore_id,
      time_processed: dayjs().utc().unix(),
   })
   await updateWebBoxscore({
      web_boxscore_id: boxscores[0].web_boxscore_id,
      time_processed: dayjs().utc().unix(),
   })
   return await processBoxScores(page);
}