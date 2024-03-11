import { dbClient } from '../../constants/dbClient.js';

export const insertWebBoxscore = async (url: string, season: number) => {
   return await dbClient.query(
      `
      INSERT INTO
        web_boxscore
         (
            url
            ,season
         )
      VALUES 
         ($1, $2)
   `,
      [
         url.trim(),
         season,
      ],
   )
}