import dayjs from 'dayjs';
import type { HTMLElement } from 'node-html-parser';
import { parse } from 'node-html-parser';
import { AtBat } from '../enums/AtBat.js';
import { Pitch } from '../enums/Pitch.js';
import type { Result } from '../interfaces/Result.js';
import type { AtBatTable } from '../interfaces/tables/AtBatTable.js';
import type { GameTable } from '../interfaces/tables/GameTable.js';
import type { PlayerTable } from '../interfaces/tables/PlayerTable.js';
import type { WebBoxscoreTable } from '../interfaces/tables/WebBoxscoreTable.js';
import { getNumber } from './getNumber.js';
import { getString } from './getString.js';
import { output } from './output.js';
import { getDBAtBats } from './queries/getDBAtBats.js';
import { getDBOldestUnprocessedBoxscore } from './queries/getDBOldestUnprocessedBoxscore.js';
import { insertDBAtBat } from './queries/insertDBAtBat.js';
import { insertDBPitch } from './queries/insertDBPitch.js';
import { updateDBWebBoxscore } from './queries/updateDBWebBoxscore.js';
import { removeDiacritics } from './removeDiacritics.js';
import { scrapeGame } from './scrapeGame.js';
import { scrapePlayer } from './scrapePlayer.js';

export const processBoxScores = async () => {
   const result: Result = {
      errors: [],
      function: 'processBoxScores()',
      messages: [],
      proceed: false,
   };

   const getAtBats = async (game: GameTable, players: PlayerTable[]) => {
      const { rows: atBats } = await getDBAtBats(game.game_id) as { rows: AtBatTable[] };
      const trs = dom.querySelectorAll('#play_by_play tbody > *');
      await Promise.all(trs.map(async tr => {
         if (result.errors.length)
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
            result.errors.push(`Could not find batter ${batterName} while getting at-bat`);
            return;
         }
         const pitcherName = removeDiacritics(tds[7].innerText.replace(/&nbsp;/g, ' '));
         const pitcher = players.find(player => player.name === pitcherName);
         if (!pitcher) {
            result.errors.push(`Could not find pitcher ${pitcherName} while getting at-bat`);
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
         let atBat: AtBat;
         if (outcome.includes('Double Play'))
            atBat = AtBat.doublePlay;
         else if (outcome.startsWith('Flyball'))
            atBat = AtBat.flyball;
         else if (outcome.startsWith('Foul Popfly'))
            atBat = AtBat.foulPopfly;
         else if (outcome.startsWith('Groundout'))
            atBat = AtBat.groundout;
         else if (outcome.startsWith('Popfly'))
            atBat = AtBat.popfly;
         else if (outcome.startsWith('Lineout'))
            atBat = AtBat.lineout;
         else if (outcome.startsWith('Strikeout'))
            atBat = AtBat.strikeout;
         else if (outcome.includes('Triple Play'))
            atBat = AtBat.triplePlay;
         else if (outcome.startsWith('Hit By Pitch'))
            atBat = AtBat.hitByPitch;
         else if (outcome.startsWith('Single to'))
            atBat = AtBat.single;
         else if (outcome.startsWith('Walk'))
            atBat = AtBat.walk;
         else if (outcome.startsWith('Double to'))
            atBat = AtBat.double;
         else if (outcome.startsWith('Ground-rule Double'))
            atBat = AtBat.groundRuleDouble;
         else if (outcome.startsWith('Triple to'))
            atBat = AtBat.triple;
         else if (outcome.startsWith('Home Run'))
            atBat = AtBat.homeRun;
         else
            atBat = AtBat.unknown;
         let fields: any = {
            bases,
            batter_player_id: batter.player_id,
            game_id: game.game_id,
            outs,
            pitcher_player_id: pitcher.player_id,
            result: atBat,
            runs,
            sequence_id: sequenceId,
            total_pitches: totalPitches,
         }
         const { rows: newAtBat } = await insertDBAtBat(fields) as { rows: AtBatTable[] }
         result.messages.push('inserted at-bat:');
         result.messages.push(fields);
         const pitches = getString(tds[3].querySelector('.pitch_sequence')?.innerText);
         [...pitches].map(async (pitch, index) => {
            let pitchType: Pitch;
            switch (pitch) {
               case 'B':
                  pitchType = Pitch.ball;
                  break;
               case 'C':
                  pitchType = Pitch.calledStrike;
                  break;
               case 'F':
                  pitchType = Pitch.foul;
                  break;
               case 'H':
                  pitchType = Pitch.hitBatter;
                  break;
               case 'I':
                  pitchType = Pitch.intentionalBall;
                  break;
               case 'K':
                  pitchType = Pitch.unknownStrike;
                  break;
               case 'L':
                  pitchType = Pitch.foulBunt;
                  break;
               case 'M':
                  pitchType = Pitch.missedBuntAttempt;
                  break;
               case 'O':
                  pitchType = Pitch.foulTipOnBunt;
                  break;
               case 'Q':
                  pitchType = Pitch.swingingOnPitchout;
                  break;
               case 'R':
                  pitchType = Pitch.foulBallOnPitchout;
                  break;
               case 'S':
                  pitchType = Pitch.swingingStrike;
                  break;
               case 'T':
                  pitchType = Pitch.foulTip;
                  break;
               case 'V':
                  pitchType = Pitch.automaticIntentionalBall;
                  break;
               case 'X':
                  pitchType = Pitch.ballPutIntoPlayByBatter;
                  break;
               case 'Y':
                  pitchType = Pitch.ballPutIntoPlayOnPitchout;
                  break;
               default:
                  pitchType = Pitch.unknownOrMissedPitch;
            }
            fields = {
               at_bat_id: newAtBat[0].at_bat_id,
               result: pitchType,
               sequence_id: index + 1,
            }
            await insertDBPitch(fields);
            result.messages.push('inserted pitch:');
            result.messages.push(fields);
         })
      }))
   }

   const getGame = async (dom: HTMLElement, url: string) => {
      const baseballReferenceId = getString(url.split('/').slice(4).join('/').split('.').shift());
      return await scrapeGame(baseballReferenceId, dom, result);
   }

   const getPlayers = async (dom: HTMLElement) => {
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
      return await scrapePlayers(baseballReferenceIds, []);
   }

   const scrapePlayers = async (baseballReferenceIds: string[], players: PlayerTable[]): Promise<PlayerTable[] | false> => {
      if (!baseballReferenceIds.length)
         return players;
      const baseballReferenceId = baseballReferenceIds.shift();
      if (!baseballReferenceId)
         return players;
      const player = await scrapePlayer(baseballReferenceId, result);
      if (result.errors.length || player === false)
         return false;
      players.push(player);
      return await scrapePlayers(baseballReferenceIds, players);
   }

   const { rows: boxscores } = await getDBOldestUnprocessedBoxscore() as { rows: WebBoxscoreTable[] };
   if (!boxscores.length) {
      result.messages.push('There are no unprocessed boxscores.');
      return output(result);
   }
   const { html, url, web_boxscore_id } = boxscores[0];
   result.messages.push(web_boxscore_id.toString());
   result.messages.push(url);
   if (!html) {
      result.errors.push('No HTML');
      return output(result);
   }
   const dom = parse(html);
   const game = await getGame(dom, url);
   if (result.errors.length || game === false)
      return output(result);
   const players = await getPlayers(dom);
   if (result.errors.length || players === false)
      return output(result);
   await getAtBats(game, players);
   if (result.errors.length)
      return output(result);
   const fields = {
      web_boxscore_id: boxscores[0].web_boxscore_id,
      time_processed: dayjs().utc().unix(),
   }
   await updateDBWebBoxscore(fields);
   result.messages.push('updated web boxscore:');
   result.messages.push(fields);
   result.proceed = true;
   return output(result);
}