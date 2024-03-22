import { dbClient } from '../../constants/dbClient.js';

export const getWebBoxscores = async () => {
   return await dbClient.query(`
      SELECT
         web_boxscore.season
         ,web_boxscore.time_retrieved
         ,web_boxscore.url
         ,web_boxscore.web_boxscore_id
      FROM
         web_boxscore
   `)
}